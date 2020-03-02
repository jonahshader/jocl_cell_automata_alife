package jonahshader

import jonahshader.opencl.CLFloatArray
import jonahshader.opencl.CLIntArray
import jonahshader.opencl.OpenCLProgram
import processing.core.PApplet
import java.util.*
import kotlin.math.pow

class Simulator(private val worldWidth: Int, private val worldHeight: Int, private val graphics: PApplet, private val numCreatures: Int, openClFilename: String, seed: Long) {
    companion object {
        const val INIT_ENERGY = 512.toShort()
        const val INIT_ENERGY_VARIANCE = 5000

        const val VISION_WIDTH_EXTEND = 4
        const val VISION_HEIGHT_EXTEND = 4
        const val VISION_LAYERS = 3 // RGB
        val VISION_SIZE = intArrayOf(VISION_WIDTH_EXTEND * 2 + 1, VISION_HEIGHT_EXTEND * 2 + 1)
        const val NN_INPUTS = ((VISION_WIDTH_EXTEND * 2 + 1) * (VISION_HEIGHT_EXTEND * 2 + 1) * VISION_LAYERS) + 1
        // output consists of actions and parameters
        // actions are: nothing, move, rotate, eat, place wall, damage, copy
        // parameters are left/right, hue x, hue y
        const val NN_OUTPUTS = 10

        val NN_HIDDEN_LAYERS = intArrayOf(30, 18, 15, 15, 15)

//        val NN_CONFIG = intArrayOf(NN_INPUTS,
//                30, 30, // hidden layers
//                NN_OUTPUT)
    }

    private val clp = OpenCLProgram(openClFilename, arrayOf("actionKernel", "actionCleanupKernel",
            "renderForegroundSimpleKernel", "renderForegroundDetailedKernel", "updateCreatureKernel", "addFoodKernel",
            "spreadFoodKernel", "flipWritingToAKernel", "renderBackgroundKernel", "spectateCreatureKernel",
            "copySpectatingToAll"))
    private var currentTick = 0L
    private var ran = Random(seed)

    // define data arrays for opencl kernels
    private val worldSize = clp.createCLIntArray(2)
    private val writingToA = clp.createCLIntArray(1)
    private val worldA = clp.createCLIntArray(worldWidth * worldHeight)
    private val worldB = clp.createCLIntArray(worldWidth * worldHeight)
    private val selectX = clp.createCLCharArray(numCreatures)
    private val selectY = clp.createCLCharArray(numCreatures)
    private val creatureX = clp.createCLIntArray(numCreatures)
    private val creatureY = clp.createCLIntArray(numCreatures)
    private val pCreatureX = clp.createCLIntArray(numCreatures)
    private val pCreatureY = clp.createCLIntArray(numCreatures)
    private val lastActionSuccess = clp.createCLCharArray(numCreatures)
    val screenSizeCenterScale = clp.createCLFloatArray(6)
    private val screen = CLIntArray(graphics.pixels, clp.context, clp.commandQueue)
    private val randomNumbers = clp.createCLIntArray(worldWidth * worldHeight)
    private val worldObjects = clp.createCLCharArray(worldWidth * worldHeight)
    private val worldFood = clp.createCLFloatArray(worldWidth * worldHeight)
    private val worldFoodBackBuffer = clp.createCLFloatArray(worldWidth * worldHeight)
    private val creatureHue = clp.createCLFloatArray(numCreatures)
    private val creatureEnergy = clp.createCLShortArray(numCreatures)
    private val creatureAction = clp.createCLCharArray(numCreatures)
    private val creatureDirection = clp.createCLCharArray(numCreatures)
    private var creatureNN: CLFloatArray
    private val creatureToSpec = clp.createCLIntArray(1)
    private val nnStructure = clp.createCLIntArray(NN_HIDDEN_LAYERS.size + 2)
    private val nnInputs = clp.createCLFloatArray(NN_INPUTS * numCreatures)
    private val visionSize = CLIntArray(VISION_SIZE, clp.context, clp.commandQueue)
    private val nnConstants = clp.createCLIntArray(2)
    private val nnOutputs = clp.createCLFloatArray(NN_OUTPUTS * numCreatures)

    init {
        var singleNNSize = 0
        // + 1 for bias neuron, + 1 for recursive weight, + 1 for recursive value storage
        if (NN_HIDDEN_LAYERS.isNotEmpty()) {
            singleNNSize += (NN_INPUTS + 3) * NN_HIDDEN_LAYERS[0]
            for (i in 0 until NN_HIDDEN_LAYERS.size - 1) {
                singleNNSize += (NN_HIDDEN_LAYERS[i] + 3) * NN_HIDDEN_LAYERS[i + 1]
            }
            singleNNSize += (NN_HIDDEN_LAYERS.last() + 3) * NN_OUTPUTS
        }
        else {
            singleNNSize += (NN_INPUTS + 3) * NN_OUTPUTS
        }
        println("Number of floats in nn: $singleNNSize")
        // now that we have the size of the nn calculated, make the CLFloatArray
        creatureNN = clp.createCLFloatArray(singleNNSize * numCreatures)


        // neural net stuff
        nnStructure.array[0] = NN_INPUTS
        for (i in NN_HIDDEN_LAYERS.indices) {
            nnStructure.array[i + 1] = NN_HIDDEN_LAYERS[i]
        }
        nnStructure.array[nnStructure.array.lastIndex] = NN_OUTPUTS
        nnConstants.array[0] = NN_HIDDEN_LAYERS.size + 1
        nnConstants.array[1] = singleNNSize

        for (i in creatureNN.array.indices) {
//            creatureNN.array[i] = ran.nextFloat() * if(ran.nextFloat() > 0.5f) 1 else -1
            creatureNN.array[i] = ran.nextGaussian().toFloat() * 1f
        }

        initWorld()

        // register data arrays with kernels
        val actionKernel = clp.getKernel("actionKernel")
        var i = 0
        worldSize.registerAndSendArgument(actionKernel, i++)
        writingToA.registerAndSendArgument(actionKernel, i++)
        worldA.registerAndSendArgument(actionKernel, i++)
        worldB.registerAndSendArgument(actionKernel, i++)
        selectX.registerAndSendArgument(actionKernel, i++)
        selectY.registerAndSendArgument(actionKernel, i++)
        creatureX.registerAndSendArgument(actionKernel, i++)
        creatureY.registerAndSendArgument(actionKernel, i++)
        pCreatureX.registerAndSendArgument(actionKernel, i++)
        pCreatureY.registerAndSendArgument(actionKernel, i++)
        lastActionSuccess.registerAndSendArgument(actionKernel, i++)
        creatureEnergy.registerAndSendArgument(actionKernel, i++)
        creatureAction.registerAndSendArgument(actionKernel, i++)
        worldObjects.registerAndSendArgument(actionKernel, i++)
        nnConstants.registerAndSendArgument(actionKernel, i++)
        creatureNN.registerAndSendArgument(actionKernel, i++)
        randomNumbers.registerAndSendArgument(actionKernel, i++)

        val actionCleanupKernel = clp.getKernel("actionCleanupKernel")
        i = 0
        worldSize.registerAndSendArgument(actionCleanupKernel, i++)
        writingToA.registerAndSendArgument(actionCleanupKernel, i++)
        worldA.registerAndSendArgument(actionCleanupKernel, i++)
        worldB.registerAndSendArgument(actionCleanupKernel, i++)
        pCreatureX.registerAndSendArgument(actionCleanupKernel, i++)
        pCreatureY.registerAndSendArgument(actionCleanupKernel, i++)

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
        selectX.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        selectY.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        creatureHue.registerAndSendArgument(renderForegroundSimpleKernel, i++)
        creatureEnergy.registerAndSendArgument(renderForegroundSimpleKernel, i++)

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
        selectX.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        selectY.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        creatureHue.registerAndSendArgument(renderForegroundDetailedKernel, i++)
        creatureEnergy.registerAndSendArgument(renderForegroundDetailedKernel, i++)

        val renderBackgroundKernel = clp.getKernel("renderBackgroundKernel")
        i = 0
        worldSize.registerAndSendArgument(renderBackgroundKernel, i++)
        screenSizeCenterScale.registerAndSendArgument(renderBackgroundKernel, i++)
        worldFood.registerAndSendArgument(renderBackgroundKernel, i++)
        worldObjects.registerAndSendArgument(renderBackgroundKernel, i++)
        screen.registerAndSendArgument(renderBackgroundKernel, i++)

        val updateCreatureKernel = clp.getKernel("updateCreatureKernel")
        i = 0
        worldSize.registerAndSendArgument(updateCreatureKernel, i++)
        writingToA.registerAndSendArgument(updateCreatureKernel, i++)
        worldA.registerAndSendArgument(updateCreatureKernel, i++)
        worldB.registerAndSendArgument(updateCreatureKernel, i++)
        selectX.registerAndSendArgument(updateCreatureKernel, i++)
        selectY.registerAndSendArgument(updateCreatureKernel, i++)
        lastActionSuccess.registerAndSendArgument(updateCreatureKernel, i++)
        randomNumbers.registerAndSendArgument(updateCreatureKernel, i++)
        creatureX.registerAndSendArgument(updateCreatureKernel, i++)
        creatureY.registerAndSendArgument(updateCreatureKernel, i++)
        creatureEnergy.registerAndSendArgument(updateCreatureKernel, i++)
        worldFood.registerAndSendArgument(updateCreatureKernel, i++)
        creatureAction.registerAndSendArgument(updateCreatureKernel, i++)
        creatureDirection.registerAndSendArgument(updateCreatureKernel, i++)
        creatureNN.registerAndSendArgument(updateCreatureKernel, i++)
        nnStructure.registerAndSendArgument(updateCreatureKernel, i++)
        nnInputs.registerAndSendArgument(updateCreatureKernel, i++)
        visionSize.registerAndSendArgument(updateCreatureKernel, i++)
        nnConstants.registerAndSendArgument(updateCreatureKernel, i++)
        creatureHue.registerAndSendArgument(updateCreatureKernel, i++)
        worldObjects.registerAndSendArgument(updateCreatureKernel, i++)
        nnOutputs.registerAndSendArgument(updateCreatureKernel, i++)

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

        val spectateCreatureKernel = clp.getKernel("spectateCreatureKernel")
        i = 0
        worldSize.registerAndSendArgument(spectateCreatureKernel, i++)
        creatureX.registerAndSendArgument(spectateCreatureKernel, i++)
        creatureY.registerAndSendArgument(spectateCreatureKernel, i++)
        pCreatureX.registerAndSendArgument(spectateCreatureKernel, i++)
        pCreatureY.registerAndSendArgument(spectateCreatureKernel, i++)
        creatureToSpec.registerAndSendArgument(spectateCreatureKernel, i++)
        screenSizeCenterScale.registerAndSendArgument(spectateCreatureKernel, i++)

        val copySpectatingToAll = clp.getKernel("copySpectatingToAll")
        i = 0
        creatureToSpec.registerAndSendArgument(copySpectatingToAll, i++)
        creatureNN.registerAndSendArgument(copySpectatingToAll, i++)
        nnStructure.registerAndSendArgument(copySpectatingToAll, i++)
        nnConstants.registerAndSendArgument(copySpectatingToAll, i++)
        nnInputs.registerAndSendArgument(copySpectatingToAll, i++)
        nnOutputs.registerAndSendArgument(copySpectatingToAll, i++)

        worldSize.copyToDevice()
        writingToA.copyToDevice()
        worldA.copyToDevice()
        worldB.copyToDevice()
        selectX.copyToDevice()
        selectY.copyToDevice()
        creatureX.copyToDevice()
        creatureY.copyToDevice()
        pCreatureX.copyToDevice()
        pCreatureY.copyToDevice()
        lastActionSuccess.copyToDevice()
        screenSizeCenterScale.copyToDevice()
        screen.copyToDevice()
        randomNumbers.copyToDevice()
        worldObjects.copyToDevice()
        worldFood.copyToDevice()
        worldFoodBackBuffer.copyToDevice()
        creatureHue.copyToDevice()
        creatureEnergy.copyToDevice()
        creatureAction.copyToDevice()
        creatureDirection.copyToDevice()
        creatureToSpec.copyToDevice()
        creatureNN.copyToDevice()
        nnStructure.copyToDevice()
        visionSize.copyToDevice()
        nnConstants.copyToDevice()
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



        // init worlds
        for (i in 0 until worldWidth * worldHeight) {
            worldA.array[i] = -1
            worldB.array[i] = -1
            randomNumbers.array[i] = ran.nextInt()
            worldFood.array[i] = ran.nextFloat().pow(8)
            worldFoodBackBuffer.array[i] = worldFood.array[i]
        }

        // init creatures
        for (i in 0 until numCreatures) {
            creatureEnergy.array[i] = (INIT_ENERGY + kotlin.math.abs(ran.nextInt()) % INIT_ENERGY_VARIANCE).toShort()

            lastActionSuccess.array[i] = 1
            creatureHue.array[i] = (ran.nextDouble() * Math.PI * 2.0).toFloat()
            creatureDirection.array[i] = (kotlin.math.abs(ran.nextInt()) % 4).toByte()

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
        clp.executeKernel("actionCleanupKernel", numCreatures.toLong())
        clp.waitForCL()
        clp.executeKernel("flipWritingToAKernel", 1L)
        clp.waitForCL()
        clp.executeKernel("updateCreatureKernel", numCreatures.toLong())
        clp.waitForCL()
        clp.executeKernel("actionKernel", numCreatures.toLong())
        clp.waitForCL()
        if (currentTick % 32 == 0L) {
            clp.executeKernel("addFoodKernel", worldWidth * worldHeight.toLong())
            clp.waitForCL()
            clp.executeKernel("spreadFoodKernel", worldWidth * worldHeight.toLong())
            clp.waitForCL()
        }

//        if (currentTick % 512 == 0L) {
//            creatureEnergy.copyFromDevice()
//            for (energy in creatureEnergy.array) {
//                if (energy < 0)
//                    println(energy)
//            }
//        }
        if (currentTick % 512 == 0L) {
//            nnOutputs.copyFromDevice()
////            for (o in nnOutputs.array)
////                println(o)

//            lastActionSuccess.copyFromDevice()
//            for (o in lastActionSuccess.array)
//                println(o)
        }
        currentTick++
    }

    fun render(xCenter: Float, yCenter: Float, zoom: Float, progress: Float, spectate: Boolean, creatureSpectating: Int) {
        assert(zoom >= 1)
        screenSizeCenterScale.array[2] = xCenter
        screenSizeCenterScale.array[3] = yCenter
        screenSizeCenterScale.array[4] = zoom
        screenSizeCenterScale.array[5] = progress
        screenSizeCenterScale.copyToDevice()
        creatureToSpec.array[0] = creatureSpectating
        creatureToSpec.copyToDevice()
        if (spectate) {
            clp.executeKernel("spectateCreatureKernel", 1)
            screenSizeCenterScale.copyFromDevice()
        }

        clp.executeKernel("renderBackgroundKernel", screen.array.size.toLong())
        clp.waitForCL()
        if (zoom > 8) {
            clp.executeKernel("renderForegroundDetailedKernel", screen.array.size.toLong())
        } else {
            clp.executeKernel("renderForegroundSimpleKernel", screen.array.size.toLong())
        }

        clp.waitForCL()
        screen.copyFromDevice()
        clp.waitForCL()
    }

    fun dispose() {
        clp.dispose()
    }

    fun replicateSpectatingToAll(creatureSpectating: Int) {
        creatureToSpec.array[0] = creatureSpectating
        creatureToSpec.copyToDevice()
        clp.executeKernel("copySpectatingToAll", numCreatures.toLong())
    }
}