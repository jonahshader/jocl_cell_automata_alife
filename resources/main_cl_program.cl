#define WHITE (0xffffffff)
#define BLACK (0xff000000)

int wrap(int value, int range);
// int indexToX(int index, global int* worldSize);
// int indexToY(int index, global int* worldSize);
int posToIndexWrapped(int x, int y, global int* worldSize);
bool isCreature(int x, int y, global int* worldSize, global int* readWorld);
int numCreaturesMovingHere(int x, int y, global int* worldSize, global int* readWorld,
  global short* moveX, global short* moveY);
bool isMovingHere(int xCell, int yCell, int xDest, int yDest, global int* worldSize,
  global int* readWorld, global short* moveX, global short* moveY);
int getCell(int x, int y, global int* worldSize, global int* readWorld);


// this kernel runs per creature
kernel void
movementKernel(global int* worldSize, global int* writingToA,
  global int* worldA, global int* worldB,
  global short* moveX, global short* moveY, global int* creatureX, global int* creatureY,
  global int* pCreatureX, global int* pCreatureY,
  global short* lastMoveSuccess)
{
  int creatureIndex = get_global_id(0);
  //copy current pos to previous pos
  pCreatureX[creatureIndex] = creatureX[creatureIndex];
  pCreatureY[creatureIndex] = creatureY[creatureIndex];

  /* figure out which world is being written to
     and which one is being read from */
  global int* readWorld;
  global int* writeWorld;
  if (writingToA[0] == 1)
  {
    writeWorld = worldA;
    readWorld = worldB;
  }
  else
  {
    writeWorld = worldB;
    readWorld = worldA;
  }

  int cx = creatureX[creatureIndex];
  int cy = creatureY[creatureIndex];
  int newX = cx;
  int newY = cy;

  // if creature is attempting to move,
  if (moveX[creatureIndex] != 0 || moveY[creatureIndex] != 0)
  {
    // check position if there is a creature there already
    int moveToX = wrap(cx + moveX[creatureIndex], worldSize[0]);
    int moveToY = wrap(cy + moveY[creatureIndex], worldSize[1]);

    int cellAtPos = readWorld[moveToX + moveToY * worldSize[0]];

    // if there is not a creature at the desired spot,
    if (cellAtPos < 0)
    {
      // check 3 neighbors to see if anyone else is trying to go there
      // if not, go there, set lastMoveSuccess to true
      // else, set lastMoveSuccess to false
      int numMovingHere = numCreaturesMovingHere(moveToX, moveToY, worldSize, readWorld, moveX, moveY);
      if (numMovingHere == 1)
      {
        // we can move successfully because we are the only one trying to go there
        // lastMoveSuccess[creatureIndex] = true;
        newX = moveToX;
        newY = moveToY;
      }
    }
  }
  writeWorld[newX + newY * worldSize[0]] = creatureIndex;
  // update position
  creatureX[creatureIndex] = newX;
  creatureY[creatureIndex] = newY;
}



// this kernel is called directly after movementKernel.
// it removes creatures from the readWorld
kernel void
movementCleanupKernel(global int* worldSize, global int* writingToA,
  global int* worldA, global int* worldB,
  global int* pCreatureX, global int* pCreatureY)
{
  int creatureIndex = get_global_id(0);

  /* figure out which world is being written to
     and which one is being read from */
  global int* readWorld = writingToA[0] ? worldB : worldA;

  // delete creatures from readWorld
  int cx = pCreatureX[creatureIndex];
  int cy = pCreatureY[creatureIndex];

  // set position in world where creature was to -1 to indicate empty space
  readWorld[cx + cy * worldSize[0]] = -1;
}

// screenSizeCenterScale is an int array of size 5. width, height, xCenter, yCenter, pixelsPerCell
kernel void
renderKernel(global int* worldSize, global int* writingToA,
  global int* worldA, global int* worldB,
  global int* screenSizeCenterScale, global int* screen)
{
  int index = get_global_id(0);
  int screenX = index % screenSizeCenterScale[0];
  int screenY = index / screenSizeCenterScale[0];

  int screenXCentered = screenX - (screenSizeCenterScale[0] / 2);
  int screenYCentered = screenY - (screenSizeCenterScale[1] / 2);

  int worldX = screenSizeCenterScale[2] + (screenXCentered / screenSizeCenterScale[4]);
  int worldY = screenSizeCenterScale[3] + (screenYCentered / screenSizeCenterScale[4]);

  /* figure out which world is being written to
     and which one is being read from */
  global int* readWorld = writingToA[0] ? worldB : worldA;

  screen[index] = readWorld[posToIndexWrapped(worldX, worldY, worldSize)] == -1 ? BLACK : WHITE;
}

inline int wrap(int value, int range)
{
  int out = value % range;
  if (out < 0) out += range;
  return out;
}

// inline int indexToX(int index, global int* worldSize)
// {
//   return index % worldSize[0];
// }
//
// inline int indexToY(int index, global int* worldSize)
// {
//   return index / worldSize[0];
// }

inline int posToIndexWrapped(int x, int y, global int* worldSize)
{
  return wrap(x, worldSize[0]) + ((wrap(y, worldSize[1]) * worldSize[0]));
}

inline bool isCreature(int x, int y, global int* worldSize, global int* readWorld)
{
  return readWorld[posToIndexWrapped(x, y, worldSize)] >= 0;
}

// assuming x y already wrapped
inline int numCreaturesMovingHere(int x, int y, global int* worldSize, global int* readWorld, global short* moveX, global short* moveY)
{
  int num = 0;
  //check top
  int xTop = x;
  int yTop = y - 1;
  num += isMovingHere(xTop, yTop, x, y, worldSize, readWorld, moveX, moveY);
  // check bottom
  int xBottom = x;
  int yBottom = y + 1;
  num += isMovingHere(xBottom, yBottom, x, y, worldSize, readWorld, moveX, moveY);
  // check left
  int xLeft = x - 1;
  int yLeft = y;
  num += isMovingHere(xLeft, yLeft, x, y, worldSize, readWorld, moveX, moveY);
  // check right
  int xRight = x + 1;
  int yRight = y;
  num += isMovingHere(xRight, yRight, x, y, worldSize, readWorld, moveX, moveY);

  return num;
}

// assuming xDest and yDest is already wrapped
inline bool isMovingHere(int xCell, int yCell, int xDest, int yDest, global int* worldSize, global int* readWorld, global short* moveX, global short* moveY)
{
  int cell = getCell(xCell, yCell, worldSize, readWorld);
  if (cell >= 0)
  {
    int creatureXDest = wrap(moveX[cell] + xCell, worldSize[0]);
    int creatureYDest = wrap(moveY[cell] + yCell, worldSize[1]);

    return ((creatureXDest == xDest) && (creatureYDest == yDest));
  }
  else
  {
    return false;
  }
}

inline int getCell(int x, int y, global int* worldSize, global int* readWorld)
{
  return readWorld[posToIndexWrapped(x, y, worldSize)];
}
