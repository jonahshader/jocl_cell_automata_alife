package jonahshader

import processing.core.PApplet
import processing.core.PConstants
import processing.event.KeyEvent
import kotlin.math.max

class App : PApplet() {
    companion object {
        const val SCREEN_WIDTH = 2560
        const val SCREEN_HEIGHT = 1440

//        const val WORLD_WIDTH = 8192/4
//        const val WORLD_HEIGHT = 8192/4
        const val WORLD_WIDTH = 2560/2
        const val WORLD_HEIGHT = 1440/2

        const val NUM_CREATURES = ((WORLD_WIDTH * WORLD_HEIGHT) / 32.0).toInt() + 1
    }

    private val noDrawToggleKey = 'o'
    private var noDrawKeyPressed = false
    private var noDraw = false

    private var leftPressed = false
    private var rightPressed = false
    private var upPressed = false
    private var downPressed = false

    private var zoom = 8.0
    private var xCam = WORLD_WIDTH / 2.0
    private var yCam = WORLD_HEIGHT / 2.0
    private var xCamVel = 0.0
    private var yCamVel = 0.0
    private var zoomVel = 0.0
    private var spectating = false
    private var creatureSpectating = 0

    private var iterationsPerFrame = 0.125f

    private lateinit var sim: Simulator

    override fun settings() {
        size(SCREEN_WIDTH, SCREEN_HEIGHT)
//        fullScreen()
        noSmooth()
//        noAlpha()

    }

    override fun exit() {
        sim.dispose()
        println("exiting...exit")
        super.exit()
    }

    override fun setup() {
        frameRate(165f)
        blendMode(PConstants.REPLACE)
        loadPixels()
        updatePixels()

        val seed = (Math.random() * Long.MAX_VALUE).toLong()

//        val numCreatures = 2049
        sim = Simulator(WORLD_WIDTH, WORLD_HEIGHT, this, NUM_CREATURES, "main_cl_program.cl", seed)
        println("seed: $seed")
        println("creatures: $NUM_CREATURES")
    }

    override fun draw() {
        if (upPressed) yCamVel -= 0.5
        if (downPressed) yCamVel += 0.5
        if (leftPressed) xCamVel -= 0.5
        if (rightPressed) xCamVel += 0.5

        xCamVel = friction(xCamVel, 0.25)
        yCamVel = friction(yCamVel, 0.25)

        xCamVel = velCap(xCamVel, 4.0)
        yCamVel = velCap(yCamVel, 4.0)

        xCam += xCamVel / zoom
        yCam += yCamVel / zoom

        xCam = wrap(xCam, WORLD_WIDTH.toDouble())
        yCam = wrap(yCam, WORLD_HEIGHT.toDouble())


        val framesPerIteration = (1 / iterationsPerFrame).toInt()

        if (iterationsPerFrame > 1) {
            for (i in 0 until iterationsPerFrame.toInt())
                sim.run()
        } else {
            if (frameCount % framesPerIteration == 0)
                sim.run()
        }


        loadPixels()
        if (iterationsPerFrame >= 1) {
            sim.render(xCam.toFloat(), yCam.toFloat(), zoom.toFloat(), 1f, spectating, creatureSpectating)
            if (spectating) {
                xCam = sim.screenSizeCenterScale.array[2].toDouble()
                yCam = sim.screenSizeCenterScale.array[3].toDouble()
            }

        } else {
            var progress = ((frameCount % framesPerIteration) / framesPerIteration.toFloat()) + iterationsPerFrame
//            progress = min(progress, 1f).pow(0.25f)
//            progress = sin(progress * PI / 2f)
//            progress = sqrt(1 - (1-progress).pow(2))
//            progress = tanh(progress * 10) / tanh(10f)
//            progress *= 5f
//            progress = (1f - cos(PI*progress))/2f
            sim.render(xCam.toFloat(), yCam.toFloat(), zoom.toFloat(), progress, spectating, creatureSpectating)
            if (spectating) {
                xCam = sim.screenSizeCenterScale.array[2].toDouble()
                yCam = sim.screenSizeCenterScale.array[3].toDouble()
            }
        }

        updatePixels()
        textAlign(LEFT, TOP)
        text("IPF: $iterationsPerFrame, TPS: ${frameRate * iterationsPerFrame}", 0f, 0f)

        red(5)
    }

    override fun keyPressed(event: KeyEvent?) {
        when (event?.key?.toLowerCase()) {
            noDrawToggleKey -> {
                noDrawKeyPressed = true
                noDraw = !noDraw
            }
            'w' -> upPressed = true
            'a' -> leftPressed = true
            's' -> downPressed = true
            'd' -> rightPressed = true
            'e' -> zoom *= 2.0
            'q' -> {
                zoom /= 2.0
                zoom = max(zoom, 1.0)
            }
            ']' -> iterationsPerFrame *= 2f
            '[' -> iterationsPerFrame *= 0.5f
            '`' -> saveFrame("screenshot.png")
            'f' -> spectating = !spectating
            '.' -> {
                creatureSpectating++
                creatureSpectating %= NUM_CREATURES
            }
            ',' -> {
                creatureSpectating--
                if (creatureSpectating < 0)
                    creatureSpectating += NUM_CREATURES
            }
            'c' -> sim.replicateSpectatingToAll(creatureSpectating)
        }
    }

    override fun keyReleased() {
        when (key.toLowerCase()) {
            noDrawToggleKey -> noDrawKeyPressed = false
            'w' -> upPressed = false
            'a' -> leftPressed = false
            's' -> downPressed = false
            'd' -> rightPressed = false
        }
    }

    private fun friction(vel: Double, reduction: Double) : Double {
        var out = vel
        if (vel > 0) {
            out -= reduction
            if (out < 0) out = 0.0
        } else {
            out += reduction
            if (out > 0) out = 0.0
        }
        return out
    }

    private fun velCap(vel: Double, cap: Double) : Double = if (vel > cap) cap else if (vel < -cap) -cap else vel
}