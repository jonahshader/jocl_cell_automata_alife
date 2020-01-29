int wrap(int value, int range);
int[] indexToPos(int index, global int* worldSize);
int posToIndexWrapped(int x, int y, global int* worldSize);
bool isCreature(int worldVal);


// this kernel runs per creature
kernel void
movementKernel(global int* worldSize, global char* writingToA,
  global int* worldA, global int* worldB,
  global char* moveX, global char* moveY, global char* lastMoveSuccess)
{
  int creature = get_global_id(0);
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
  if (moveX[creature] != 0 || moveY[creature] != 0)
  {
    // check position if there is a creature there already
    int moveToPos[2] = indexToPos();
    moveToPos[0] += moveX[creature];
    moveToPos[1] += moveY[creature];

    int cellAtPos = readWorld[posToIndexWrapped(moveToPos[0], moveToPos[1], worldSize)];

    // if there is not a creature at the desired spot,
    if (!isCreature(cellAtPos))
    {
      // check 3 neighbors to see if anyone else is trying to to there
      // if not, go there, set lastMoveSuccess to true
      // else, set lastMoveSuccess to false
    }
  }

}

inline int wrap(int value, int range)
{
  int out = value % range;
  if (out < 0) out += range;
  return out;
}

inline int[] indexToPos(int index, global int* worldSize)
{
  int out[2];
  out[0] = index % worldSize[0];
  out[1] = index / worldSize[0];
}

inline int posToIndexWrapped(int x, int y, global int* worldSize)
{
  return wrap(x, worldSize[0]) + wrap(y, worldSize[1]) * worldSize[0];
}

inline bool isCreature(int worldValue)
{
  return worldValue >= 0;
}
