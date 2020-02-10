package jonahshader

import jonahshader.opencl.CLIntArray
import jonahshader.opencl.OpenCLProgram
import processing.core.PApplet
import java.util.*

class Simulator(private val worldWidth: Int, private val worldHeight: Int, private val graphics: PApplet, private val numCreatures: Int, openClFilename: String, seed: Long) {
    companion object {
        const val INIT_ENERGY = 128.toShort()
    }

    private val clp = OpenCLProgram(openClFilename, arrayOf("movementKernel", "movementCleanupKernel",
            "renderForegroundSimpleKernel", "renderForegroundDetailedKernel", "updateCreatureKernel", "addFoodKernel",
            "spreadFoodKernel", "flipWritingToAKernel", "renderBackgroundKernel"))
    private var currentTick = 0L
    private var ran = Random(seed)

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
    private val screenSizeCenterScale = clp.createCLFloatArray(6)
    private val screen = CLIntArray(graphics.pixels, clp.context, clp.commandQueue)
    private val randomNumbers = clp.createCLIntArray(worldWidth * worldHeight)
    private val worldObjects = clp.createCLShortArray(worldWidth * worldHeight)
    private val worldFood = clp.createCLFloatArray(worldWidth * worldHeight)
    private val worldFoodBackBuffer = clp.createCLFloatArray(worldWidth * worldHeight)
    private val creatureHue = clp.createCLFloatArray(numCreatures)
    private val creatureEnergy = clp.createCLShortArray(numCreatures)

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
        creatureEnergy.registerAndSendArgument(movementKernel, i++)

        val movementCleanupKernel = clp.getKernel("movementCleanupKernel")
        i = 0
        worldSize.registerAndSendArgument(movementCleanupKernel, i++)
        writingToA.registerAndSendArgument(movementCleanupKernel, i++)
        worldA.registerAndSendArgument(movementCleanupKernel, i++)
        worldB.registerAndSendArgument(movementCleanupKernel, i++)
        pCreatureX.registerAndSendArgument(movementCleanupKernel, i++)
        pCreatureY.registerAndSendArgument(movementCleanupKernel, i++)

        val renderForegroundSimpleKernel = clp.getKernel("renderForegroundSimpleKernel")
        i = 0
        worldSize.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        writingToA.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        worldA.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        worldB.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        creatureX.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        creatureY.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        pCreatureX.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        pCreatureY.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        screenSizeCenterScale.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        screen.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        moveX.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        moveY.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        creatureHue.registerAndSendArgument(renderForegroundSimpleKernel, i++)

        val renderForegroundDetailedKernel = clp.getKernel("renderForegroundDetailedKernel")
        i = 0
        worldSize.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        writingToA.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        worldA.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        worldB.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        creatureX.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        creatureY.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        pCreatureX.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        pCreatureY.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        screenSizeCenterScale.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        screen.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        moveX.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        moveY.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        creatureHue.registerAndSendArgument(renderForegroundDetailedKernel, i++)

        val renderBackgroundKernel = clp.getKernel("renderBackgroundKernel")
        i = 0
        worldSize.registerAndSendArgument(renderBackgroundKernel, i++)
        screenSizeCenterScale.registerAndSendArgument(renderBackgroundKernel, i++)
        worldFood.registerAndSendArgument(renderBackgroundKernel, i++)
        screen.registerAndSendArgument(renderBackgroundKernel, i++)

        val updateCreatureKernel = clp.getKernel("updateCreatureKernel")
        i = 0
        worldSize.registerAndSendArgument(updateCreatureKernel, i++)
        writingToA.registerAndSendArgument(updateCreatureKernel, i++)
        worldA.registerAndSendArgument(updateCreatureKernel, i++)
        worldB.registerAndSendArgument(updateCreatureKernel, i++)
        moveX.registerAndSendArgument(updateCreatureKernel, i++)
        moveY.registerAndSendArgument(updateCreatureKernel, i++)
        lastMoveSuccess.registerAndSendArgument(updateCreatureKernel, i++)
        randomNumbers.registerAndSendArgument(updateCreatureKernel, i++)
        creatureX.registerAndSendArgument(updateCreatureKernel, i++)
        creatureY.registerAndSendArgument(updateCreatureKernel, i++)
        creatureEnergy.registerAndSendArgument(updateCreatureKernel, i++)
        worldFood.registerAndSendArgument(updateCreatureKernel, i++)

        val addFoodKernel = clp.getKernel("addFoodKernel")
        i = 0
        worldSize.registerAndSendArgument(addFoodKernel, i++)
        worldFood.registerAndSendArgument(addFoodKernel, i++)
        worldFoodBackBuffer.registerAndSendArgument(addFoodKernel, i++)
        randomNumbers.registerAndSendArgument(addFoodKernel, i++)

        val spreadFoodKernel = clp.getKernel("spreadFoodKernel")
        i = 0
        worldSize.registerAndSendArgument(spreadFoodKernel, i++)
        worldFood.registerAndSendArgument(spreadFoodKernel, i++)
        worldFoodBackBuffer.registerAndSendArgument(spreadFoodKernel, i++)
        randomNumbers.registerAndSendArgument(spreadFoodKernel, i++)

        val flipWritingToAKernel = clp.getKernel("flipWritingToAKernel")
        i = 0
        writingToA.registerAndSendArgument(flipWritingToAKernel, i++)

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
        randomNumbers.copyToDevice()
        worldObjects.copyToDevice()
        worldFood.copyToDevice()
        worldFoodBackBuffer.copyToDevice()
        creatureHue.copyToDevice()
        creatureEnergy.copyToDevice()
    }

    private fun initWorld() {
        worldSize.array[0] = worldWidth
        worldSize.array[1] = worldHeight

        screenSizeCenterScale.array[0] = graphics.width.toFloat()
        screenSizeCenterScale.array[1] = graphics.height.toFloat()
        screenSizeCenterScale.array[2] = 0f
        screenSizeCenterScale.array[3] = 1f
        screenSizeCenterScale.array[4] = 1f
        screenSizeCenterScale.array[5] = 1f

        // init worlds to -1
        for (i in 0 until worldWidth * worldHeight) {
            worldA.array[i] = -1
            worldB.array[i] = -1
            randomNumbers.array[i] = ran.nextInt()
        }

        // init creatures
        for (i in 0 until numCreatures) {
            var tempMoveX = 0
            var tempMoveY = 0
            val polarity = ran.nextFloat() > 0.5
            val direction = ran.nextFloat() > 0.5
            if (polarity)
                tempMoveX = if (direction) 1 else -1
            else
                tempMoveY = if (direction) 1 else -1

            moveX.array[i] = tempMoveX.toShort()
            moveY.array[i] = tempMoveY.toShort()
            creatureEnergy.array[i] = INIT_ENERGY

            lastMoveSuccess.array[i] = 1
            creatureHue.array[i] = (ran.nextDouble() * Math.PI * 2.0).toFloat()

            var findingSpotForCreature = true
            while (findingSpotForCreature) {
                val x = (ran.nextFloat() * worldWidth).toInt()
                val y = (ran.nextFloat() * worldHeight).toInt()

                if (worldA.array[x + y * worldWidth] == -1) {
                    creatureX.array[i] = x
                    creatureY.array[i] = y
                    pCreatureX.array[i] = x
                    pCreatureY.array[i] = y
                    worldA.array[x + y * worldWidth] = i
                    worldB.array[x + y * worldWidth] = i
                    findingSpotForCreature = false
                }
            }
        }
    }

    fun run() {
        clp.executeKernel("movementCleanupKernel", numCreatures.toLong())
        clp.waitForCL()
        clp.executeKernel("flipWritingToAKernel", 1L)
        clp.waitForCL()
        clp.executeKernel("updateCreatureKernel", numCreatures.toLong())
        clp.waitForCL()
        clp.executeKernel("movementKernel", numCreatures.toLong())
        clp.waitForCL()
        if (currentTick % 32 == 0L) {
            clp.executeKernel("addFoodKernel", worldWidth * worldHeight.toLong())
            clp.executeKernel("spreadFoodKernel", worldWidth * worldHeight.toLong())
        }
        currentTick++
    }

    fun render(xCenter: Float, yCenter: Float, zoom: Float, progress: Float) {
        assert(zoom >= 1)
        screenSizeCenterScale.array[2] = xCenter
        screenSizeCenterScale.array[3] = yCenter
        screenSizeCenterScale.array[4] = zoom
        screenSizeCenterScale.array[5] = progress
        screenSizeCenterScale.copyToDevice()
        clp.executeKernel("renderBackgroundKernel", graphics.width * graphics.height.toLong())
        if (zoom > 8) {
            clp.executeKernel("renderForegroundDetailedKernel", (graphics.width * graphics.height).toLong())
        } else {
            clp.executeKernel("renderForegroundSimpleKernel", (graphics.width * graphics.height).toLong())
        }

        clp.waitForCL()
        screen.copyFromDevice()
        clp.waitForCL()
    }
}