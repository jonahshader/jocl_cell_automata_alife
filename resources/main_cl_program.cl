int wrap(int value, int range);
int indexToX(int index, global int* worldSize);
int indexToY(int index, global int* worldSize);
int posToIndexWrapped(int x, int y, global int* worldSize);
bool isCreature(int worldVal);
bool isCreature(int x, int y, global int* worldSize);
int numCreaturesMovingHere(int x, int y, global int* worldSize, global int* readWorld,
  global char* moveX, global char* moveY);
bool isMovingHere(int xCell, int yCell, int xDest, int yDest, global int* worldSize,
  global int* readWorld, global char* moveX, global char* moveY);
int getCell(int x, int y, global int* worldSize, global int* readWorld);


// this kernel runs per creature
kernel void
movementKernel(global int* worldSize, global char* writingToA,
  global int* worldA, global int* worldB,
  global char* moveX, global char* moveY, global int* creatureX, global int* creatureY,
  global char* lastMoveSuccess)
{
  int creatureIndex = get_global_id(0);
  /* figure out which world is being written to
     and which one is being read from */
  global const int* readWorld, writeWorld;
  if (writingToA[0])
  {
    writeWorld = worldA;
    readWorld = worldB;
  }
  else
  {
    writeWorld = worldB;
    readWorld = worldA;
  }

  // if creature is attempting to move,
  if (moveX[creatureIndex] != 0 || moveY[creatureIndex] != 0)
  {
    // check position if there is a creature there already
    int moveToX = indexToX(creatureX[creatureIndex], worldSize) + moveX[creatureIndex];
    int moveToY = indexToY(creatureY[creatureIndex], worldSize) + moveY[creatureIndex];

    int cellAtPos = readWorld[posToIndexWrapped(moveToX, moveToY, worldSize)];

    // if there is not a creature at the desired spot,
    if (!isCreature(cellAtPos))
    {
      // check 3 neighbors to see if anyone else is trying to to there
      // if not, go there, set lastMoveSuccess to true
      // else, set lastMoveSuccess to false
      int numMovingHere = numCreaturesMovingHere(moveToX, moveToY, worldSize, readWorld, moveX, moveY);
      if (numMovingHere == 1)
      {
        // we can move successfully because we are the only one trying to go there
        lastMoveSuccess[creatureIndex] = true;
        writeWorld[posToIndexWrapped(moveToX, moveToY)] = creatureIndex;
        creatureX[creatureIndex] = moveToX;
        creatureY[creatureIndex] = moveToY;
      }
      else
      {
        // we cannot move successfully, conflict present
        lastMoveSuccess[creatureIndex] = false;
        //writeWorld[] TODO:
      }
    }
  }
}

kernel void
movementCleanupKernel(global int* worldSize, global char* writingToA,
  global int* worldA, global int* worldB,
  global int* creatureX, global int* creatureY, global char* lastMoveSuccess)
{
  int creatureIndex = get_global_id(0);
  if (lastMoveSuccess[])
}

inline int wrap(int value, int range)
{
  int out = value % range;
  if (out < 0) out += range;
  return out;
}

inline int indexToX(int index, global int* worldSize)
{
  return index % worldSize[0];
}

inline int indexToY(int index, global int* worldSize)
{
  return index / worldSize[0];
}

inline int posToIndexWrapped(int x, int y, global int* worldSize)
{
  return wrap(x, worldSize[0]) + wrap(y, worldSize[1]) * worldSize[0];
}

inline bool isCreature(int worldValue)
{
  return worldValue >= 0;
}

inline bool isCreature(int x, int y, global int* worldSize)
{
  return isCreature(posToIndexWrapped(x, y, worldSize));
}

// assuming x y already wrapped
inline int numCreaturesMovingHere(int x, int y, global int* worldSize, global int* readWorld, global char* moveX, global char* moveY)
{
  int num = 0;
  //check top
  int xTop = x;
  int yTop = y - 1;
  num += isMovingHere(x, y, xTop, yTop, worldSize, readWorld, moveX, moveY);
  // check bottom
  int xBottom = x;
  int yBottom = y + 1;
  num += isMovingHere(x, y, xBottom, yBottom, worldSize, readWorld, moveX, moveY);
  // check left
  int xLeft = x - 1;
  int yLeft = y;
  num += isMovingHere(x, y, xLeft, yLeft, worldSize, readWorld, moveX, moveY);
  // check right
  int xRight = x + 1;
  int yRight = y;
  num += isMovingHere(x, y, xRight, yRight, worldSize, readWorld, moveX, moveY);

  return num;
}

// todo: optimize out unnessesary wrapping
inline bool isMovingHere(int xCell, int yCell, int xDest, int yDest, global int* worldSize, global int* readWorld, global char* moveX, global char* moveY)
{
  int cell = getCell(xCell, yCell, worldSize, readWorld);
  if (isCreature(cell))
  {
    int creatureXDest = wrap(moveX[cell] + xCell, worldSize[0]);
    int creatureYDest = wrap(moveY[cell] + yCell;, worldSize[1]);
    int xDestWrapped = wrap(xDest, worldSize[0]);
    int yDestWrapped = wrap(yDest, worldSize[1]);

    return (creatureXDest == xDestWrapped && creatureYDest == yDestWrapped);
  }
  return false;
}

inline int getCell(int x, int y, global int* worldSize, global int* readWorld)
{
  return readWorld[posToIndexWrapped(x, y, worldSize)];
}
