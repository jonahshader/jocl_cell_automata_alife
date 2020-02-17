

/*
actions are: do nothing, move forward/backward, rotate,
eat, place wall, harm/remove wall, copy genetics,
*/
typedef enum {
  NOTHING,
  MOVE,
  ROTATE,
  EAT,
  PLACE_WALL,
  DAMAGE,
  COPY,
  NUM_ACTIONS
} Action;



// uncomment option to enable
// #define ANIMATION_STEPPING_ENABLED

// one out of every ADD_FOOD_CHANCE will gain food
// when the addFoodKernel runs
#define ADD_FOOD_CHANCE (256)

#define ENERGY_PER_FOOD (128)
#define WALL_PLACE_ENERGY_COST (1)
#define WALL_REMOVE_ENERGY_COST (3)

#define WALL_RED (100)
#define WALL_GREEN (50)
#define WALL_BLUE (70)

// dont touch these
#define HUE_SPACING (2.09439510239319549231f)
#define WHITE (0xffffffff)
#define BLACK (0xff000000)

int wrap(int value, int range);
// int indexToX(int index, global int* worldSize);
// int indexToY(int index, global int* worldSize);
int posToIndexWrapped(int x, int y, global int* worldSize);
bool isCreature(int x, int y, global int* worldSize, global int* readWorld);
int numCreaturesSelectingHere(int x, int y, global int* worldSize, global int* readWorld,
  global char* selectX, global char* selectY);
bool isMovingHere(int xCell, int yCell, int xDest, int yDest, global int* worldSize,
  global int* readWorld, global char* selectX, global char* selectY);
int getCell(int x, int y, global int* worldSize, global int* readWorld);
int roundEven(float number);
float interpolate(float a, float b, float progress);
unsigned int getNextRandom(int index, global unsigned int* randomNumbers);
int numNeighbors(int x, int y, global int* readWorld, global int* worldSize);
float fwrap(float value, float range);
float hueToRed(float hue);
float hueToGreen(float hue);
float hueToBlue(float hue);
int rgbToRed(int rgb);
int rgbToGreen(int rgb);
int rgbToBlue(int rgb);
int componentsToRgb(int red, int green, int blue);
void eat(int creature, int foodIndex, global short* creatureEnergy, global float* food);
void updateCreatureSelection(int creature, global char* creatureDirection, global char* selectX, global char* selectY);
// works on x or y
float getCreaturePosInterp(int creature, global int* pCreatureLoc, global int* creatureLoc, float progress, int axisSize);

void nnForwardProp(int creature, global int* nnStructure, global float* creaturenn);

kernel void
updateCreatureKernel(global int* worldSize, global int* writingToA,
  global int* worldA, global int* worldB,
  global char* selectX, global char* selectY,
  global char* lastActionSuccess, global unsigned int* randomNumbers,
  global int* creatureX, global int* creatureY, global short* creatureEnergy,
  global float* worldFood, global char* creatureAction, global char* creatureDirection,
  global float* creaturenn,
  global int* nnStructure // note: nnStructure's first element is the number of layers (n). the following n elements are the sizes of each layer (without bias neuron and all that stuff)
  )
{
  int creature = get_global_id(0);
  global int* readWorld = writingToA[0] ? worldB : worldA;
  global int* writeWorld = writingToA[0] ? worldA : worldB;
  int x = creatureX[creature];
  int y = creatureY[creature];

  int worldIndex = posToIndexWrapped(x, y, worldSize);

  selectX[creature] = 0;
  selectY[creature] = 0;
  if (creatureEnergy[creature] >= 1) // if creature is alive,
  {
    // determine next action
    int nextAction = getNextRandom(creature, randomNumbers) % NUM_ACTIONS;
    creatureAction[creature] = nextAction;
    int nextDirection = (getNextRandom(creature, randomNumbers) % 2) * 2 - 1;
    switch (nextAction)
    {
      case MOVE:
        updateCreatureSelection(creature, creatureDirection, selectX, selectY);
        break;
      case ROTATE:
        creatureDirection[creature] = wrap(creatureDirection[creature] + nextDirection, 4);
        break;
      case EAT:
        eat(creature, worldIndex, creatureEnergy, worldFood);
        break;
      case PLACE_WALL:
        updateCreatureSelection(creature, creatureDirection, selectX, selectY);
        break;
      case DAMAGE:
        updateCreatureSelection(creature, creatureDirection, selectX, selectY);
        break;
      case COPY:
        updateCreatureSelection(creature, creatureDirection, selectX, selectY);
        break;
      default:
      case NOTHING:
        // :D
        break;
    }
  }
  else
  {
    creatureAction[creature] = NOTHING;
  }
}


// this kernel runs per creature
kernel void
actionKernel(global int* worldSize, global int* writingToA,
  global int* worldA, global int* worldB,
  global char* selectX, global char* selectY, global int* creatureX, global int* creatureY,
  global int* pCreatureX, global int* pCreatureY,
  global char* lastActionSuccess, global short* creatureEnergy, global char* creatureAction,
  global char* worldObjects)
{
  int creature = get_global_id(0);
  //copy current pos to previous pos
  pCreatureX[creature] = creatureX[creature];
  pCreatureY[creature] = creatureY[creature];

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

  int cx = creatureX[creature];
  int cy = creatureY[creature];
  int newX = cx;
  int newY = cy;

  bool actionSuccessful = false;

  // if creature is attempting to do something
  if (selectX[creature] != 0 || selectY[creature] != 0)
  {
    // check position if there is a creature there already
    int selectPosX = wrap(cx + selectX[creature], worldSize[0]);
    int selectPosY = wrap(cy + selectY[creature], worldSize[1]);

    int selectIndex = selectPosX + selectPosY * worldSize[0];

    int cellAtPos = readWorld[selectIndex];
    int objAtPos = worldObjects[selectIndex];

    // if there is not a creature at the desired spot,
    // check 3 neighbors to see if anyone else is trying to select there
    int numSelectingHere = numCreaturesSelectingHere(selectPosX, selectPosY, worldSize, readWorld, selectX, selectY);
    if (numSelectingHere == 1)
    {
      switch (creatureAction[creature])
      {
        case MOVE:
          if (objAtPos == 0 && cellAtPos == -1)
          {
            newX = selectPosX;
            newY = selectPosY;
            actionSuccessful = true;
            creatureEnergy[creature]--;
          }
          break;
        case ROTATE:
          creatureEnergy[creature]--;
          actionSuccessful = true;
          break;
        case PLACE_WALL:
          if (objAtPos == 0 && cellAtPos == -1 && creatureEnergy[creature] >= WALL_PLACE_ENERGY_COST)
          {
            worldObjects[selectIndex] = 1; // for wall
            actionSuccessful = true;
            creatureEnergy[creature] -= WALL_PLACE_ENERGY_COST;
          }
          break;
        case DAMAGE:
          if (cellAtPos == -1 && creatureEnergy[creature] >= WALL_REMOVE_ENERGY_COST)
          {
            worldObjects[selectIndex] = 0; // remove whatever is there
            actionSuccessful = true;
            creatureEnergy[creature] -=WALL_REMOVE_ENERGY_COST;
          }
          break;
        case COPY:
          if (cellAtPos != -1)
          {
            if (creatureEnergy[cellAtPos] == 0 && creatureEnergy[creature] > 1)
            {
              creatureEnergy[cellAtPos] += creatureEnergy[creature] / 2;
              creatureEnergy[creature] /= 2;
              actionSuccessful = true;
              creatureEnergy[creature]--;
              // if (creatureEnergy[creature] < 0)
              //   creatureEnergy[creature] = 0;
            }
          }
          break;
        default:
          actionSuccessful = true;
          break;
      }
    }
  }
  writeWorld[newX + newY * worldSize[0]] = creature;
  // update position
  creatureX[creature] = newX;
  creatureY[creature] = newY;
  lastActionSuccess[creature] = actionSuccessful;
}


// this kernel is called directly after movementKernel.
// it removes creatures from the readWorld
kernel void
actionCleanupKernel(global int* worldSize, global int* writingToA,
  global int* worldA, global int* worldB,
  global int* pCreatureX, global int* pCreatureY)
{
  int creature = get_global_id(0);

  /* figure out which world is being written to
     and which one is being read from */
  global int* readWorld = writingToA[0] ? worldB : worldA;

  // delete creatures from readWorld
  int cx = pCreatureX[creature];
  int cy = pCreatureY[creature];

  // set position in world where creature was to -1 to indicate empty space
  readWorld[cx + cy * worldSize[0]] = -1;
}

// screenSizeCenterScale is an int array of size 5. width, height, xCenter, yCenter, pixelsPerCell, progress (0 to 1)
kernel void
renderForegroundDetailedKernel(global int* worldSize, global int* writingToA,
  global int* worldA, global int* worldB,
  global int* creatureX, global int* creatureY,
  global int* pCreatureX, global int* pCreatureY,
  global float* screenSizeCenterScale, global int* screen,
  global char* selectX, global char* selectY,
  global float* creatureHue, global short* creatureEnergy)
{
  int index = get_global_id(0);
  int screenX = index % ((int)screenSizeCenterScale[0]);
  int screenY = index / ((int)screenSizeCenterScale[0]);

  float screenXCentered = screenX - (screenSizeCenterScale[0] / 2.0f);
  float screenYCentered = screenY - (screenSizeCenterScale[1] / 2.0f);

  float worldXF = screenSizeCenterScale[2] + (screenXCentered / screenSizeCenterScale[4]);
  float worldYF = screenSizeCenterScale[3] + (screenYCentered / screenSizeCenterScale[4]);

  worldXF = fwrap(worldXF + .5f, worldSize[0])-.5f;
  worldYF = fwrap(worldYF + .5f, worldSize[1])-.5f;

  int worldX = wrap(floor(worldXF + .5f), worldSize[0]);
  int worldY = wrap(floor(worldYF + .5f), worldSize[1]);

  /* figure out which world is being written to
     and which one is being read from */
  global int* readWorld = writingToA[0] ? worldB : worldA;
  global int* writeWorld = writingToA[0] ? worldA : worldB;

  int cell = readWorld[posToIndexWrapped(worldX, worldY, worldSize)];
  if (cell < 0) cell = writeWorld[posToIndexWrapped(worldX, worldY, worldSize)];

  int red = 0;
  int green = 0;
  int blue = 0;
  if (cell >= 0)
  {
    int pcx = pCreatureX[cell];
    int pcy = pCreatureY[cell];
    int cx = creatureX[cell];
    int cy = creatureY[cell];

    // undo wrapping from pCreature to creature for correct interpolation
    // if this wasn't here, the interpolation would make the creature fly
    // across the map because on the previous frame it could have been
    // on x = 0, and next frame x = -1 which gets wrapped to world width - 1
    // interpolating between 0 and world width - 1 would make the creature
    // fly across the map, which is not the intended behavior
    if (abs(pcx - cx) > 1)
    {
      if (worldX > (worldSize[0]/2))
      {
        if (pcx < cx) pcx += worldSize[0];
        else          cx  += worldSize[0];
      }
      else
      {
        if (pcx < cx) cx  -= worldSize[0];
        else          pcx -= worldSize[0];
      }
    }
    else if (abs(pcy - cy) > 1)
    {
      if (worldY > (worldSize[1]/2))
      {
        if (pcy < cy) pcy += worldSize[1];
        else          cy  += worldSize[1];
      }
      else
      {
        if (pcy < cy) cy  -= worldSize[1];
        else          pcy -= worldSize[1];
      }
    }

    // interpolate position of creature moving from previous pos to new pos
    float customProgress;
    #ifdef ANIMATION_STEPPING_ENABLED
    customProgress = pow(screenSizeCenterScale[5], 2.0f);
    #endif // ANIMATION_STEPPING_ENABLED
    #ifndef ANIMATION_STEPPING_ENABLED
    customProgress = screenSizeCenterScale[5];
    #endif // ANIMATION_STEPPING_ENABLED
    float ix = interpolate(pcx, cx, customProgress);
    float iy = interpolate(pcy, cy, customProgress);
    float dx = ix - worldXF;
    float dy = iy - worldYF;

    // renderAsSquares means the camera is zoomed out too far for circles to make screenSizeCenterScale
    // so they will just be rendered as squares
    if (creatureEnergy[cell] == 0)
    {
      red = 180;
      green = 200;
      blue = 200;
    }
    else if (creatureEnergy[cell] < 0)
    {
      red = 255;
      green = 0;
      blue = 0;
    }
    else
    {
      float hue = creatureHue[cell];
      red = hueToRed(hue) * 255.0f;
      green = hueToGreen(hue) * 255.0f;
      blue = hueToBlue(hue) * 255.0f;
    }


    float distToCreatureCenter = sqrt(dx * dx + dy * dy);
    float brightness;
    #ifdef ANIMATION_STEPPING_ENABLED
    float progMult = (pow((0.5f - customProgress) * 2, 2.0f) + 1)/2.0f;
    if (pcx == cx && pcy == cy) progMult = 1.0f;
    float maxDist = 1/progMult;
    float maxDistPow = maxDist * maxDist;
    float customDist = distToCreatureCenter / progMult;
    brightness = max(0.5f - customDist, 0.0f) * 2.0f;
    brightness = sqrt(1-pow(brightness - 1, 2.0f)) * progMult;
    #endif // ANIMATION_STEPPING_ENABLED
    #ifndef ANIMATION_STEPPING_ENABLED
    brightness = max(0.5f - distToCreatureCenter, 0.0f) * 2.0f;
    brightness = sqrt(1-pow(brightness - 1, 2.0f));
    #endif // ANIMATION_STEPPING_ENABLED

    red *= brightness * .5f + .5f;
    green *= brightness * .5f + .5f;
    blue *= brightness * .5f + .5f;
    int screenRgb = screen[index];
    int screenRed = rgbToRed(screenRgb);
    int screenGreen = rgbToGreen(screenRgb);
    int screenBlue = rgbToBlue(screenRgb);
    float partCreaturePartScreen = pow(brightness, 0.75f);
    int newRed = interpolate(screenRed, red, partCreaturePartScreen);
    int newGreen = interpolate(screenGreen, green, partCreaturePartScreen);
    int newBlue = interpolate(screenBlue, blue, partCreaturePartScreen);

    screen[index] = componentsToRgb(newRed, newGreen, newBlue);
  }
}

kernel void
renderForegroundSimpleKernel(global int* worldSize, global int* writingToA,
  global int* worldA, global int* worldB,
  global int* creatureX, global int* creatureY,
  global int* pCreatureX, global int* pCreatureY,
  global float* screenSizeCenterScale, global int* screen,
  global char* selectX, global char* selectY,
  global float* creatureHue, global short* creatureEnergy)
{
  int index = get_global_id(0);
  int screenX = index % ((int)screenSizeCenterScale[0]);
  int screenY = index / ((int)screenSizeCenterScale[0]);

  float screenXCentered = screenX - (screenSizeCenterScale[0] / 2.0f);
  float screenYCentered = screenY - (screenSizeCenterScale[1] / 2.0f);

  float worldXF = screenSizeCenterScale[2] + (screenXCentered / screenSizeCenterScale[4]);
  float worldYF = screenSizeCenterScale[3] + (screenYCentered / screenSizeCenterScale[4]);

  worldXF = fwrap(worldXF + .5f, worldSize[0])-.5f;
  worldYF = fwrap(worldYF + .5f, worldSize[1])-.5f;

  int worldX = wrap(floor(worldXF + .5f), worldSize[0]);
  int worldY = wrap(floor(worldYF + .5f), worldSize[1]);

  /* figure out which world is being written to
     and which one is being read from */
  global int* readWorld = writingToA[0] ? worldB : worldA;
  global int* writeWorld = writingToA[0] ? worldA : worldB;

  int cell = readWorld[posToIndexWrapped(worldX, worldY, worldSize)];
  if (cell < 0) cell = writeWorld[posToIndexWrapped(worldX, worldY, worldSize)];

  int red = 0;
  int green = 0;
  int blue = 0;
  if (cell >= 0)
  {
    int pcx = pCreatureX[cell];
    int pcy = pCreatureY[cell];
    int cx = creatureX[cell];
    int cy = creatureY[cell];

    // undo wrapping from pCreature to creature for correct interpolation
    // if this wasn't here, the interpolation would make the creature fly
    // across the map because on the previous frame it could have been
    // on x = 0, and next frame x = -1 which gets wrapped to world width - 1
    // interpolating between 0 and world width - 1 would make the creature
    // fly across the map, which is not the intended behavior
    if (abs(pcx - cx) > 1)
    {
      if (worldX > (worldSize[0]/2))
      {
        if (pcx < cx) pcx += worldSize[0];
        else          cx  += worldSize[0];
      }
      else
      {
        if (pcx < cx) cx  -= worldSize[0];
        else          pcx -= worldSize[0];
      }
    }
    else if (abs(pcy - cy) > 1)
    {
      if (worldY > (worldSize[1]/2))
      {
        if (pcy < cy) pcy += worldSize[1];
        else          cy  += worldSize[1];
      }
      else
      {
        if (pcy < cy) cy  -= worldSize[1];
        else          pcy -= worldSize[1];
      }
    }

    // interpolate position of creature moving from previous pos to new pos
    float customProgress = screenSizeCenterScale[5];
    float ix = interpolate(pcx, cx, customProgress);
    float iy = interpolate(pcy, cy, customProgress);

    // find square boundaries
    float xMin = ix - 0.5f;
    float xMax = ix + 0.5f;
    float yMin = iy - 0.5f;
    float yMax = iy + 0.5f;

    bool renderSquare = worldXF >= xMin && worldXF <= xMax && worldYF >= yMin && worldYF <= yMax;
    if (renderSquare)
    {
      if (creatureEnergy[cell] == 0)
      {
        red = 180;
        green = 200;
        blue = 200;
      }
      else
      {
        float hue = creatureHue[cell];
        red = hueToRed(hue) * 255.0f;
        green = hueToGreen(hue) * 255.0f;
        blue = hueToBlue(hue) * 255.0f;
      }
      screen[index] = componentsToRgb(red, green, blue);
    }
  }
}

kernel void
renderBackgroundKernel(global int* worldSize,
  global float* screenSizeCenterScale,
  global float* worldFood, global char* worldObjects, global int* screen)
{
  int index = get_global_id(0);

  int screenX = index % ((int)screenSizeCenterScale[0]);
  int screenY = index / ((int)screenSizeCenterScale[0]);

  float screenXCentered = screenX - (screenSizeCenterScale[0] / 2.0f);
  float screenYCentered = screenY - (screenSizeCenterScale[1] / 2.0f);

  float worldXF = screenSizeCenterScale[2] + (screenXCentered / screenSizeCenterScale[4]);
  float worldYF = screenSizeCenterScale[3] + (screenYCentered / screenSizeCenterScale[4]);

  worldXF = fwrap(worldXF + .5f, worldSize[0])-.5f;
  worldYF = fwrap(worldYF + .5f, worldSize[1])-.5f;

  int worldX = wrap(floor(worldXF + .5f), worldSize[0]);
  int worldY = wrap(floor(worldYF + .5f), worldSize[1]);

  int worldIndex = posToIndexWrapped(worldX, worldY, worldSize);
  float foodValue = worldFood[worldIndex];
  bool drawWall = worldObjects[worldIndex];
  if (drawWall)
  {
    screen[index] = componentsToRgb(WALL_RED, WALL_GREEN, WALL_BLUE);
  }
  else
  {
    int green = foodValue * 255;
    green = min(255, green);
    screen[index] = componentsToRgb(0, green, 0);
  }
}

// runs per cell
kernel void
addFoodKernel(global int* worldSize, global float* worldFood,
  global float* worldFoodBackBuffer,
  global unsigned int* randomNumbers)
{
  int index = get_global_id(0);

  // copy current to back buffer
  worldFoodBackBuffer[index] = worldFood[index];

  if ((getNextRandom(index, randomNumbers) % ADD_FOOD_CHANCE) == 0)
  {
    worldFoodBackBuffer[index] = min(1.0f, worldFoodBackBuffer[index] + 0.5f);
  }
}

kernel void
spreadFoodKernel(global int* worldSize, global float* worldFood,
  global float* worldFoodBackBuffer,
  global unsigned int* randomNumbers)
{
  int index = get_global_id(0);
  int x = index % worldSize[0];
  int y = index / worldSize[0];

  float blurResult = 0.0f;
  for (int iy = -1; iy <= 1; iy++)
  {
    for (int ix = -1; ix <= 1; ix++)
    {
      blurResult += worldFoodBackBuffer[posToIndexWrapped(x + ix, y + iy, worldSize)];
    }
  }
  blurResult /= 9.0f;
  worldFood[index] = interpolate(blurResult, worldFoodBackBuffer[posToIndexWrapped(x, y, worldSize)], 0.5f);
}

kernel void
flipWritingToAKernel(global int* writingToA)
{
  writingToA[0] = writingToA[0] == 0 ? 1 : 0;
}

kernel void
spectateCreatureKernel(global int* worldSize, global int* creatureX, global int* creatureY,
  global int* pCreatureX, global int* pCreatureY,
  global int* creatureToSpec, global float* screenSizeCenterScale)
{
  int c = creatureToSpec[0];
  float prog = screenSizeCenterScale[5];
  screenSizeCenterScale[2] = getCreaturePosInterp(c, pCreatureX, creatureX, prog, worldSize[0]);
  screenSizeCenterScale[3] = getCreaturePosInterp(c, pCreatureY, creatureY, prog, worldSize[1]);
}

inline int wrap(int value, int range)
{
  int out = value % range;
  if (out < 0) out += range;
  return out;
}

inline float fwrap(float value, float range)
{
  float out = fmod(value, range);
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
inline int numCreaturesSelectingHere(int x, int y, global int* worldSize, global int* readWorld, global char* selectX, global char* selectY)
{
  int num = 0;
  //check top
  int xTop = x;
  int yTop = y - 1;
  num += isMovingHere(xTop, yTop, x, y, worldSize, readWorld, selectX, selectY);
  // check bottom
  int xBottom = x;
  int yBottom = y + 1;
  num += isMovingHere(xBottom, yBottom, x, y, worldSize, readWorld, selectX, selectY);
  // check left
  int xLeft = x - 1;
  int yLeft = y;
  num += isMovingHere(xLeft, yLeft, x, y, worldSize, readWorld, selectX, selectY);
  // check right
  int xRight = x + 1;
  int yRight = y;
  num += isMovingHere(xRight, yRight, x, y, worldSize, readWorld, selectX, selectY);

  return num;
}

// assuming xDest and yDest is already wrapped
inline bool isMovingHere(int xCell, int yCell, int xDest, int yDest, global int* worldSize, global int* readWorld, global char* selectX, global char* selectY)
{
  int cell = getCell(xCell, yCell, worldSize, readWorld);
  if (cell >= 0)
  {
    int creatureXDest = wrap(selectX[cell] + xCell, worldSize[0]);
    int creatureYDest = wrap(selectY[cell] + yCell, worldSize[1]);

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

inline int numNeighbors(int x, int y, global int* readWorld, global int* worldSize)
{
  int neighbors = 0;

  neighbors += readWorld[posToIndexWrapped(x - 1, y - 1, worldSize)] >= 0;
  neighbors += readWorld[posToIndexWrapped(x - 0, y - 1, worldSize)] >= 0;
  neighbors += readWorld[posToIndexWrapped(x + 1, y - 1, worldSize)] >= 0;

  neighbors += readWorld[posToIndexWrapped(x - 1, y - 0, worldSize)] >= 0;
  neighbors += readWorld[posToIndexWrapped(x + 1, y - 0, worldSize)] >= 0;

  neighbors += readWorld[posToIndexWrapped(x - 1, y + 1, worldSize)] >= 0;
  neighbors += readWorld[posToIndexWrapped(x - 0, y + 1, worldSize)] >= 0;
  neighbors += readWorld[posToIndexWrapped(x + 1, y + 1, worldSize)] >= 0;

  return neighbors;
}

inline float hueToRed(float hue)
{
  return (1.0f + cos(hue)) / 2.0f;
}

inline float hueToGreen(float hue)
{
  return (1.0f + cos(hue + HUE_SPACING)) / 2.0f;
}

inline float hueToBlue(float hue)
{
  return (1.0f + cos(hue + HUE_SPACING * 2.0f)) / 2.0f;
}

inline int rgbToRed(int rgb){return (rgb >> 16) & 0xff;}
inline int rgbToGreen(int rgb){return (rgb >> 8) & 0xff;}
inline int rgbToBlue(int rgb){return rgb & 0xff;}

inline int componentsToRgb(int red, int green, int blue)
{
  return 0xff000000 | (red << 16) | (green << 8) | (blue);
}

inline void eat(int creature, int worldIndex, global short* creatureEnergy, global float* worldFood)
{
  short newFood = worldFood[worldIndex] * ENERGY_PER_FOOD + creatureEnergy[creature];
  // check for overflow
  if (newFood > 0)
  {
    creatureEnergy[creature] = newFood;
    worldFood[worldIndex] = 0.0f;
  }
}

inline void updateCreatureSelection(int creature, global char* creatureDirection, global char* selectX, global char* selectY)
{
  switch (creatureDirection[creature])
  {
    case 0: // down
      selectY[creature] = 1;
      break;
    case 1: // right
      selectX[creature] = 1;
      break;
    case 2: // up
      selectY[creature] = -1;
      break;
    case 3: // left
      selectX[creature] = -1;
      break;
    default:
      break;
  }
}

inline int rotatePointX(int xOrigin, int yOrigin, int xPoint, int yPoint, char direction)
{
  switch (direction)
  {
    default:
    case 0: // down
      return xPoint;
    case 1: // right
      return xOrigin + (yPoint - yOrigin);
    case 2: // up
      return xOrigin - (xPoint - xOrigin);
    case 3: // left
      return xOrigin - (yPoint - yOrigin);
  }
}

inline int rotatePointY(int xOrigin, int yOrigin, int xPoint, int yPoint, char direction)
{
  switch (direction)
  {
    default:
    case 0:
      return yPoint;
    case 1:
      return yOrigin + (xPoint - xOrigin);
    case 2:
      return yOrigin - (yPoint - yOrigin);
    case 3:
      return yOrigin - (xPoint - xOrigin);
  }
}

inline float getCreaturePosInterp(int creature, global int* pCreatureLoc, global int* creatureLoc, float progress, int axisSize)
{
  int pc = pCreatureLoc[creature];
  int c = creatureLoc[creature];

  if (abs(pc - c) > 1)
  {
    if (pc < c) pc += axisSize;
    else        c  += axisSize;
  }

  return interpolate(pc, c, progress);
}

//TODO: need somewhere to store input values for neural network
inline void nnForwardProp(int creature, global int* nnStructure, global float* creaturenn)
{
  int nnSize = 0;
  for (int i = 0; i < nnStructure[0] - 1; i++)
  {
    nnSize += (nnStructure[i] + 3) * nnStructure[i + 1];
  }
  int nnStartIndex = creature * nnSize;

  // loop through all layers
  for (int i = 0; i < nnStructure[0] - 1; i++)
  {
    float sum = 0.0f;
  }
}
