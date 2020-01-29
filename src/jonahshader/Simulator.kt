package jonahshader

import jonahshader.opencl.CLIntArray
import jonahshader.opencl.OpenCLProgram

class Simulator(private val worldWidth: Int, private val worldHeight: Int, private val numCreatures: Int, openClFilename: String) {
    private val clp = OpenCLProgram(openClFilename, arrayOf("movementKernel", "movementCleanupKernel"))

    private var localViewUpdated = false

    // define data arrays for opencl kernels
    val worldSize = clp.createCLIntArray(2)
    val writingToA = clp.createCLCharArray(1)
    val worldA = clp.createCLIntArray(worldWidth * worldHeight)
    val worldB = clp.createCLIntArray(worldWidth * worldHeight)
    val moveX = clp.createCLCharArray(numCreatures)
    val moveY = clp.createCLCharArray(numCreatures)
    val creatureX = clp.createCLIntArray(numCreatures)
    val creatureY = clp.createCLIntArray(numCreatures)
    val pCreatureX = clp.createCLIntArray(numCreatures)
    val pCreatureY = clp.createCLIntArray(numCreatures)
    val lastMoveSuccess = clp.createCLCharArray(numCreatures)

    init {
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
    }

    private fun initWorld() {
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


        }
    }

    fun run() {
        clp.executeKernel("movementKernel", numCreatures.toLong())
        clp.executeKernel("movementCleanupKernel", numCreatures.toLong())

        if (writingToA.array[0].toInt() == 0)
            writingToA.array[0] = 1.toChar()
        else
            writingToA.array[0] = 0.toChar()

        writingToA.copyToDevice()
        localViewUpdated = false
    }

    fun getUpdatedWorld() : IntArray {
        if (!localViewUpdated) {
            if (writingToA.array[0].toInt() == 0)
                worldA.copyFromDevice()
            else
                worldB.copyFromDevice()
            localViewUpdated = true
        }

        return if (writingToA.array[0].toInt() == 0)
            worldA.array
        else
            worldB.array
    }

}