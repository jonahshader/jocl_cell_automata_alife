package jonahshader

import jonahshader.opencl.CLFloatArray
import jonahshader.opencl.CLIntArray
import jonahshader.opencl.OpenCLProgram
import processing.core.PApplet
import java.util.*
import kotlin.math.pow

class Simulator(private val worldWidth: Int, private val worldHeight: Int, private val graphics: PApplet, private val numCreatures: Int, openClFilename: String, seed: Long) {
    companion object {
        const val INIT_ENERGY = 200.toShort()
        const val INIT_ENERGY_VARIANCE = 6000

        const val VISION_WIDTH_EXTEND = 2
        const val VISION_HEIGHT_EXTEND = 2
        const val VISION_LAYERS = 3 // RGB
        val VISION_SIZE = intArrayOf(VISION_WIDTH_EXTEND * 2 + 1, VISION_HEIGHT_EXTEND * 2 + 1)
        // +2 at the end is for lastActionSuccess and food level
        const val NN_INPUTS = ((VISION_WIDTH_EXTEND * 2 + 1) * (VISION_HEIGHT_EXTEND * 2 + 1) * VISION_LAYERS) + 2
        // output consists of actions and parameters
        // actions are: nothing, move, rotate, eat, place wall, damage, copy
        // parameters are left/right, hue x, hue y
        const val NN_OUTPUTS = 10

//        val NN_LAYERS = intArrayOf(NN_INPUTS, 15, 15, 15, 15, 15, 15, 15, NN_OUTPUTS)
        val NN_LAYERS = intArrayOf(NN_INPUTS, 20, 18, 18, 18, 15, NN_OUTPUTS)
//        val NN_LAYERS = intArrayOf(NN_INPUTS, 17, 17, 17, 17, 17, 17, NN_OUTPUTS)


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
    private val screen = CLIntArray(graphics.pixels, clp.context, clp.commandQueue, clp)
    private val randomNumbers = clp.createCLIntArray(worldWidth * worldHeight)
    private val worldObjects = clp.createCLCharArray(worldWidth * worldHeight)
    private val worldFood = clp.createCLFloatArray(worldWidth * worldHeight)
    private val worldFoodBackBuffer = clp.createCLFloatArray(worldWidth * worldHeight)
    private val creatureHue = clp.createCLFloatArray(numCreatures)
    private val creatureEnergy = clp.createCLShortArray(numCreatures)
    private val creatureAction = clp.createCLCharArray(numCreatures)
    private val creatureDirection = clp.createCLCharArray(numCreatures)
    private val creatureToSpec = clp.createCLIntArray(1)
    private val visionSize = CLIntArray(VISION_SIZE, clp.context, clp.commandQueue, clp)

    private val weightsPerNN = clp.createCLIntArray(1)
    private val neuronsPerNN = clp.createCLIntArray(1)
    private val nnStructure = CLIntArray(NN_LAYERS, clp.context, clp.commandQueue, clp)
    private val nnNumLayers = clp.createCLIntArray(1)

    private lateinit var nnWeights: CLFloatArray
    private lateinit var nnNeuronOutputs: CLFloatArray
    private var weightArrayLayerStartIndices = clp.createCLIntArray(nnStructure.array.size)
    private var neuronArrayLayerStartIndices = clp.createCLIntArray(nnStructure.array.size)

    init {
        initNeuralNetworks()
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
        randomNumbers.registerAndSendArgument(actionKernel, i++)
        weightsPerNN.registerAndSendArgument(actionKernel, i++)
        nnWeights.registerAndSendArgument(actionKernel, i++)
        neuronsPerNN.registerAndSendArgument(actionKernel, i++)
        nnNeuronOutputs.registerAndSendArgument(actionKernel, i++)
        creatureDirection.registerAndSendArgument(actionKernel, i++)

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
        visionSize.registerAndSendArgument(updateCreatureKernel, i++)
        creatureHue.registerAndSendArgument(updateCreatureKernel, i++)
        worldObjects.registerAndSendArgument(updateCreatureKernel, i++)
        nnNeuronOutputs.registerAndSendArgument(updateCreatureKernel, i++)
        neuronsPerNN.registerAndSendArgument(updateCreatureKernel, i++)
        weightsPerNN.registerAndSendArgument(updateCreatureKernel, i++)
        nnStructure.registerAndSendArgument(updateCreatureKernel, i++)
        nnNumLayers.registerAndSendArgument(updateCreatureKernel, i++)
        weightArrayLayerStartIndices.registerAndSendArgument(updateCreatureKernel, i++)
        neuronArrayLayerStartIndices.registerAndSendArgument(updateCreatureKernel, i++)
        nnWeights.registerAndSendArgument(updateCreatureKernel, i++)

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
        visionSize.copyToDevice()

        weightsPerNN.copyToDevice()
        neuronsPerNN.copyToDevice()
        nnStructure.copyToDevice()
        nnNumLayers.copyToDevice()

        nnWeights.copyToDevice()
        nnNeuronOutputs.copyToDevice()
        weightArrayLayerStartIndices.copyToDevice()
        neuronArrayLayerStartIndices.copyToDevice()
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
            worldFood.array[i] = 0.01f + ran.nextFloat().pow(8) * 0.1f
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

    private fun initNeuralNetworks() {
        nnNumLayers.array[0] = nnStructure.array.size
        weightArrayLayerStartIndices.array[0] = 0
        for (i in 1 until nnStructure.array.size) {
            weightsPerNN.array[0] += (nnStructure.array[i - 1] + 2) * (nnStructure.array[i])
            weightArrayLayerStartIndices.array[i] = weightsPerNN.array[0]
        }

        for (i in nnStructure.array.indices) {
            neuronArrayLayerStartIndices.array[i] = neuronsPerNN.array[0]
            neuronsPerNN.array[0] += nnStructure.array[i]
        }

        nnNeuronOutputs = clp.createCLFloatArray(numCreatures * neuronsPerNN.array[0])
        nnWeights = clp.createCLFloatArray(numCreatures * weightsPerNN.array[0])

        // init weights
        for (i in nnWeights.array.indices) {
            nnWeights.array[i] = (ran.nextFloat() * 2 - 1).toFloat()
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
//            creatureDirection.copyFromDevice()
//            worldA.copyFromDevice()
//            worldB.copyFromDevice()
//            var creatureCount = 0
//            for (i in creatureEnergy.array) {
//                if (i > 0) creatureCount++
//            }
//            println("Num living creatures: $creatureCount")
//
//            val directions = HashMap<Byte, Int>()
//            for (i in creatureDirection.array) {
//                if (directions.containsKey(i)) {
//                    directions[i] = directions[i]!! + 1
//                } else {
//                    directions[i] = 1
//                }
//            }
//
//            for (i in directions.keys) {
//                println("$i direction: ${directions[i]}")
//            }
//
//            var num0 = 0
//            for (i in worldA.array) {
//                if (i == 0) num0++
//            }
//            println(num0)
//
//            num0 = 0
//            for (i in worldB.array) {
//                if (i == 0)num0++
//            }
//            println(num0)
//        }


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
        if (zoom > 4) {
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