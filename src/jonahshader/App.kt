package jonahshader

import processing.core.PApplet
import processing.event.KeyEvent
import kotlin.math.pow
import kotlin.math.tanh

class App : PApplet() {
    companion object {
        const val SCREEN_WIDTH = 800
        const val SCREEN_HEIGHT = 600

        const val WORLD_WIDTH = 8192/32
        const val WORLD_HEIGHT = 8192/32
    }

    private val noDrawToggleKey = 'o'
    private var noDrawKeyPressed = false
    private var noDraw = false

    private var leftPressed = false
    private var rightPressed = false
    private var upPressed = false
    private var downPressed = false

    private var zoom = 8f
    private var xCam = WORLD_WIDTH / 2f
    private var yCam = WORLD_HEIGHT / 2f

    private var iterationsPerFrame = 0.125f

    private lateinit var sim: Simulator

    override fun settings() {
        size(SCREEN_WIDTH, SCREEN_HEIGHT)
        noSmooth()
    }

    override fun setup() {
        frameRate(60f)

        loadPixels()
        updatePixels()

        sim = Simulator(WORLD_WIDTH, WORLD_HEIGHT, this, (WORLD_WIDTH * WORLD_HEIGHT) / 4, "main_cl_program.cl")
    }

    override fun draw() {
        if (upPressed) yCam -= 8/zoom
        if (downPressed) yCam += 8/zoom
        if (leftPressed) xCam -= 8/zoom
        if (rightPressed) xCam += 8/zoom

        val framesPerIteration = (1 / iterationsPerFrame).toInt()

        if (iterationsPerFrame > 1) {
            for (i in 0 until iterationsPerFrame.toInt())
                sim.run()
        } else {
            if (frameCount % framesPerIteration == 0)
                sim.run()
        }


        loadPixels()
        if (iterationsPerFrame > 1) {
            sim.render(xCam, yCam, zoom, 1f)
        } else {
            var progress = ((frameCount % framesPerIteration) / framesPerIteration.toFloat()) + iterationsPerFrame
//            progress = min(progress, 1f).pow(0.25f)
//            progress = sin(progress * PI / 2f)
//            progress = sqrt(1 - (1-progress).pow(2))
//            progress = tanh(progress * 10) / tanh(10f)
//            progress *= 5f
//            progress = (1f - cos(PI*progress))/2f
            sim.render(xCam, yCam, zoom, progress)
        }

        updatePixels()
        textAlign(LEFT, TOP)
        text("IPF: $iterationsPerFrame, FPS: $frameRate", 0f, 0f)
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
            'e' -> zoom *= 2f
            'q' -> {
                zoom /= 2f
                zoom = max(zoom, 1f)
            }
            ']' -> iterationsPerFrame *= 2f
            '[' -> iterationsPerFrame *= 0.5f
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
}