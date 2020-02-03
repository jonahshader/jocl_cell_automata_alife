#define WHITE (0xffffffff)
#define BLACK (0xff000000)

#define ADD_FOOD_CHANCE 50

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
int roundEven(float number);
float interpolate(float a, float b, float progress);
unsigned int getNextRandom(int index, global unsigned int* randomNumbers);


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

  bool moveSuccessful = false;

  // if creature is attempting to move,
  if (moveX[creatureIndex] != 0 || moveY[creatureIndex] != 0)
  {
    // check position if there is a creature there already
    int moveToX = wrap(cx + moveX[creatureIndex], worldSize[0]);
    int moveToY = wrap(cy + moveY[creatureIndex], worldSize[1]);

    int cellAtPos = readWorld[moveToX + moveToY * worldSize[0]];

    // if there is not a creature at the desired spot,
    if (cellAtPos == -1)
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
        moveSuccessful = true;
      }
    }
  }
  writeWorld[newX + newY * worldSize[0]] = creatureIndex;
  // update position
  creatureX[creatureIndex] = newX;
  creatureY[creatureIndex] = newY;
  lastMoveSuccess[creatureIndex] = moveSuccessful;
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

// screenSizeCenterScale is an int array of size 5. width, height, xCenter, yCenter, pixelsPerCell, progress (0 to 1)
kernel void
renderKernel(global int* worldSize, global int* writingToA,
  global int* worldA, global int* worldB,
  global int* creatureX, global int* creatureY,
  global int* pCreatureX, global int* pCreatureY,
  global float* screenSizeCenterScale, global int* screen)
{
  int index = get_global_id(0);
  int screenX = index % ((int)screenSizeCenterScale[0]);
  int screenY = index / ((int)screenSizeCenterScale[0]);

  float screenXCentered = screenX - (screenSizeCenterScale[0] / 2.0f);
  float screenYCentered = screenY - (screenSizeCenterScale[1] / 2.0f);

  float worldXF = screenSizeCenterScale[2] + (screenXCentered / screenSizeCenterScale[4]);
  float worldYF = screenSizeCenterScale[3] + (screenYCentered / screenSizeCenterScale[4]);

  // int worldX = floor(worldXF);
  // int worldY = floor(worldYF);
  int worldX = worldXF + 0.5f;
  int worldY = worldYF + 0.5f;

  /* figure out which world is being written to
     and which one is being read from */
  global int* readWorld = writingToA[0] ? worldB : worldA;
  global int* writeWorld = writingToA[0] ? worldA : worldB;

  int cell = readWorld[posToIndexWrapped(worldX, worldY, worldSize)];
  if (cell < 0) cell = writeWorld[posToIndexWrapped(worldX, worldY, worldSize)];

  int color = BLACK;
  if (cell >= 0)
  {
    float cx = interpolate(pCreatureX[cell], creatureX[cell], screenSizeCenterScale[5]);
    float cy = interpolate(pCreatureY[cell], creatureY[cell], screenSizeCenterScale[5]);
    float xMin = cx - 0.5f;
    float xMax = cx + 0.5f;
    float yMin = cy - 0.5f;
    float yMax = cy + 0.5f;
    float dx = cx - worldXF;
    float dy = cy - worldYF;
    float dist = sqrt(dx * dx + dy * dy);
    if (dist <= 0.5f)
    // if (worldXF >= xMin && worldXF <= xMax && worldYF >= yMin && worldYF <= yMax)
    {
      int colorType = cell % 3;
      if (colorType == 0)
        color = 0xffffbbbb;
      else if (colorType == 1)
        color = 0xffbbffbb;
      else
        color = 0xffbbbbff;
    }
  }
  else if (cell == -2)
  {
    color = 0xffccffcc;
  }
  screen[index] = color;
}

// runs per cell
kernel void
addFoodKernel(global int* worldSize, global int* writingToA,
  global int* worldA, global int* worldB,
  global unsigned int* randomNumbers)
{
  int index = get_global_id(0);

  if (worldA[index] == -1 && worldB[index] == -1)
  {
    if ((getNextRandom(index, randomNumbers) % ADD_FOOD_CHANCE) == 0)
    {
      worldA[index] = -2;
      worldB[index] = -2;
    }
  }
}

kernel void
updateCreatureKernel(global int* worldSize, global int* writingToA,
  global int* worldA, global int* worldB,
  global short* moveX, global short* moveY, global short* lastMoveSuccess, global unsigned int* randomNumbers)
{
  int creature = get_global_id(0);
  if (!lastMoveSuccess[creature])
  {
    // moveX[creature] = -moveX[creature];
    // moveY[creature] = -moveY[creature];
    short mx = moveX[creature];
    short my = moveY[creature];

    // moveX[creature] = my;
    // moveY[creature] = mx;

    // if (mx == 1 && my == 0)
    // {
    //   mx = 0;
    //   my = 1;
    // }
    // else if (mx == 0 && my == 1)
    // {
    //   mx = -1;
    //   my = 0;
    // }
    // else if (mx == -1 && my == 0)
    // {
    //   mx = 0;
    //   my = -1;
    // }
    // else
    // {
    //   mx = 1;
    //   my = 0;
    // }

    // mx = -mx;
    // my = -my;

    unsigned int ranNum = getNextRandom(creature, randomNumbers) % 4;

    mx = 0;
    my = 0;

    if (ranNum == 0)
    {
      mx = 1;
    }
    else if (ranNum == 1)
    {
      my = 1;
    }
    else if (ranNum == 2)
    {
      mx = -1;
    }
    else
    {
      my = -1;
    }

    moveX[creature] = mx;
    moveY[creature] = my;
  }
}

inline int wrap(int value, int range)
{
  int out = value % range;
  if (out < 0) out += range;
  return out;
}

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

inline int roundEven(float number) {
   int sign = (int)((number > 0) - (number < 0));
   int odd = ((int)number % 2); // odd -> 1, even -> 0
   return ((int)(number-sign*(0.5f-odd)));
}

inline float interpolate(float a, float b, float progress)
{
  return ((1-progress) * a) + (progress * b);
}

inline unsigned int getNextRandom(int index, global unsigned int* randomNumbers)
{
  unsigned int x = randomNumbers[index];
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  randomNumbers[index] = x;
  return x;
}
