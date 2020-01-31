package jonahshader

import jonahshader.opencl.CLIntArray
import jonahshader.opencl.OpenCLProgram

class Simulator(private val worldWidth: Int, private val worldHeight: Int, private val numCreatures: Int, openClFilename: String) {
    private val clp = OpenCLProgram(openClFilename, arrayOf("movementKernel", "movementCleanupKernel"))

    private var localViewUpdated = false

    // define data arrays for opencl kernels
    private val worldSize = clp.createCLIntArray(2)
    private val writingToA = clp.createCLIntArray(1)
    private val worldA = clp.createCLShortArray(worldWidth * worldHeight)
    private val worldB = clp.createCLShortArray(worldWidth * worldHeight)
    private val moveX = clp.createCLShortArray(numCreatures)
    private val moveY = clp.createCLShortArray(numCreatures)
    private val creatureX = clp.createCLIntArray(numCreatures)
    private val creatureY = clp.createCLIntArray(numCreatures)
    private val pCreatureX = clp.createCLIntArray(numCreatures)
    private val pCreatureY = clp.createCLIntArray(numCreatures)
    private val lastMoveSuccess = clp.createCLShortArray(numCreatures)

    init {
        initWorld()

        // register data arrays with kernels
        val movementKernel = clp.getKernel("movementKernel")
        var i = 0
        worldSize.registerAndSendArgument(movementKernel, i++)
        writingToA.registerAndSendArgument(movementKernel, i++)
        worldA.registerAndSendArgument(movementKernel, i++)
        worldB.registerAndSendArgument(movementKernel, i++)
        moveX.registerAndSendArgument(movementKernel, i++)
        moveY.registerAndSendArgument(movementKernel, i++)
        creatureX.registerAndSendArgument(movementKernel, i++)
        creatureY.registerAndSendArgument(movementKernel, i++)
        pCreatureX.registerAndSendArgument(movementKernel, i++)
        pCreatureY.registerAndSendArgument(movementKernel, i++)
        lastMoveSuccess.registerAndSendArgument(movementKernel, i++)

        val movementCleanupKernel = clp.getKernel("movementCleanupKernel")
        i = 0
        worldSize.registerAndSendArgument(movementCleanupKernel, i++)
        writingToA.registerAndSendArgument(movementCleanupKernel, i++)
        worldA.registerAndSendArgument(movementCleanupKernel, i++)
        worldB.registerAndSendArgument(movementCleanupKernel, i++)
        pCreatureX.registerAndSendArgument(movementCleanupKernel, i++)
        pCreatureY.registerAndSendArgument(movementCleanupKernel, i++)

        worldSize.copyToDevice()
        writingToA.copyToDevice()
        worldA.copyToDevice()
        worldB.copyToDevice()
        moveX.copyToDevice()
        moveY.copyToDevice()
        creatureX.copyToDevice()
        creatureY.copyToDevice()
        pCreatureX.copyToDevice()
        pCreatureY.copyToDevice()
        lastMoveSuccess.copyToDevice()
    }

    private fun initWorld() {
        worldSize.array[0] = worldWidth
        worldSize.array[1] = worldHeight

        // init worlds to -1
        for (i in 0 until worldWidth * worldHeight) {
            worldA.array[i] = -1
            worldB.array[i] = -1
        }
        // init creatures
        for (i in 0 until numCreatures) {
            var tempMoveX = 0
            var tempMoveY = 0
            val polarity = Math.random() > 0.5
            val direction = Math.random() > 0.5
            if (polarity)
                tempMoveX = if (direction) 1 else -1
            else
                tempMoveY = if (direction) 1 else -1

            moveX.array[i] = tempMoveX.toShort()
            moveY.array[i] = tempMoveY.toShort()

            var findingSpotForCreature = true
            while (findingSpotForCreature) {
                val x = (Math.random() * worldWidth).toInt()
                val y = (Math.random() * worldHeight).toInt()

                if (worldA.array[x + y * worldWidth].toInt() == -1) {
                    creatureX.array[i] = x
                    creatureY.array[i] = y
                    pCreatureX.array[i] = x
                    pCreatureY.array[i] = y
                    worldA.array[x + y * worldWidth] = i.toShort()
//                    worldB.array[x + y * worldWidth] = i.toShort()
                    findingSpotForCreature = false
                }
            }
        }
    }

    fun run() {
        clp.executeKernel("movementKernel", numCreatures.toLong())
        clp.executeKernel("movementCleanupKernel", numCreatures.toLong())

        if (writingToA.array[0] == 0)
            writingToA.array[0] = 1
        else
            writingToA.array[0] = 0

        writingToA.copyToDevice()
        localViewUpdated = false
    }

    fun getUpdatedWorld() : ShortArray {
        if (!localViewUpdated) {
            if (writingToA.array[0] == 0)
                worldA.copyFromDevice()
            else
                worldB.copyFromDevice()
            localViewUpdated = true
        }

        return if (writingToA.array[0] == 0)
            worldA.array
        else
            worldB.array
    }

}