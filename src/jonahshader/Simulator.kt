package jonahshader

import jonahshader.opencl.CLIntArray
import jonahshader.opencl.OpenCLProgram
import processing.core.PApplet

class Simulator(private val worldWidth: Int, private val worldHeight: Int, private val graphics: PApplet, private val numCreatures: Int, openClFilename: String) {
    private val clp = OpenCLProgram(openClFilename, arrayOf("movementKernel", "movementCleanupKernel", "renderKernel", "updateCreatureKernel"))

    private var localViewUpdated = false

    // define data arrays for opencl kernels
    private val worldSize = clp.createCLIntArray(2)
    private val writingToA = clp.createCLIntArray(1)
    private val worldA = clp.createCLIntArray(worldWidth * worldHeight)
    private val worldB = clp.createCLIntArray(worldWidth * worldHeight)
    private val moveX = clp.createCLShortArray(numCreatures)
    private val moveY = clp.createCLShortArray(numCreatures)
    private val creatureX = clp.createCLIntArray(numCreatures)
    private val creatureY = clp.createCLIntArray(numCreatures)
    private val pCreatureX = clp.createCLIntArray(numCreatures)
    private val pCreatureY = clp.createCLIntArray(numCreatures)
    private val lastMoveSuccess = clp.createCLShortArray(numCreatures)
    private val screenSizeCenterScale = clp.createCLIntArray(5)
    private val screen = CLIntArray(graphics.pixels, clp.context, clp.commandQueue)

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

        val renderKernel = clp.getKernel("renderKernel")
        i = 0
        worldSize.registerAndSendArgument(renderKernel, i++)
        writingToA.registerAndSendArgument(renderKernel, i++)
        worldA.registerAndSendArgument(renderKernel, i++)
        worldB.registerAndSendArgument(renderKernel, i++)
        screenSizeCenterScale.registerAndSendArgument(renderKernel, i++)
        screen.registerAndSendArgument(renderKernel, i++)

        val updateCreatureKernel = clp.getKernel("updateCreatureKernel")
        i = 0
        worldSize.registerAndSendArgument(updateCreatureKernel, i++)
        writingToA.registerAndSendArgument(updateCreatureKernel, i++)
        worldA.registerAndSendArgument(updateCreatureKernel, i++)
        worldB.registerAndSendArgument(updateCreatureKernel, i++)
        moveX.registerAndSendArgument(updateCreatureKernel, i++)
        moveY.registerAndSendArgument(updateCreatureKernel, i++)
        lastMoveSuccess.registerAndSendArgument(updateCreatureKernel, i++)

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
        screenSizeCenterScale.copyToDevice()
        screen.copyToDevice()
    }

    private fun initWorld() {
        worldSize.array[0] = worldWidth
        worldSize.array[1] = worldHeight

        screenSizeCenterScale.array[0] = graphics.width
        screenSizeCenterScale.array[1] = graphics.height
        screenSizeCenterScale.array[2] = 0
        screenSizeCenterScale.array[3] = 1
        screenSizeCenterScale.array[4] = 1

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

            lastMoveSuccess.array[i] = 1

            var findingSpotForCreature = true
            while (findingSpotForCreature) {
                val x = (Math.random() * worldWidth).toInt()
                val y = (Math.random() * worldHeight).toInt()

                if (worldA.array[x + y * worldWidth] == -1) {
                    creatureX.array[i] = x
                    creatureY.array[i] = y
                    pCreatureX.array[i] = x
                    pCreatureY.array[i] = y
                    worldA.array[x + y * worldWidth] = i
                    findingSpotForCreature = false
                }
            }
        }
    }

    fun run() {
        clp.executeKernel("updateCreatureKernel", numCreatures.toLong())
        clp.waitForCL()
        clp.executeKernel("movementKernel", numCreatures.toLong())
        clp.waitForCL()
        clp.executeKernel("movementCleanupKernel", numCreatures.toLong())
        clp.waitForCL()

        if (writingToA.array[0] == 0)
            writingToA.array[0] = 1
        else
            writingToA.array[0] = 0

        writingToA.copyToDevice()
        localViewUpdated = false
        clp.waitForCL()
    }

    fun render(xCenter: Int, yCenter: Int, zoom: Int) {
        assert(zoom >= 1)
        screenSizeCenterScale.array[2] = xCenter
        screenSizeCenterScale.array[3] = yCenter
        screenSizeCenterScale.array[4] = zoom
        screenSizeCenterScale.copyToDevice()
        clp.executeKernel("renderKernel", (graphics.width * graphics.height).toLong())
        clp.waitForCL()
        screen.copyFromDevice()
        clp.waitForCL()
    }

    fun getUpdatedWorld() : IntArray {
        if (!localViewUpdated) {
            if (writingToA.array[0] == 0)
                worldA.copyFromDevice()
            else
                worldB.copyFromDevice()
            clp.waitForCL()
            localViewUpdated = true
        }

        return if (writingToA.array[0] == 0)
            worldA.array
        else
            worldB.array
    }

}