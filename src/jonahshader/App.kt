package jonahshader

import processing.core.PApplet
import processing.core.PConstants
import processing.event.KeyEvent

class App : PApplet() {
    companion object {
        const val SCREEN_WIDTH = 1024
        const val SCREEN_HEIGHT = 1024

        const val WORLD_WIDTH = 8192 / 4
        const val WORLD_HEIGHT = 8192 / 4
    }

    private val noDrawToggleKey = 'o'
    private var noDrawKeyPressed = false
    private var noDraw = false

    private var leftPressed = false
    private var rightPressed = false
    private var upPressed = false
    private var downPressed = false

    private var zoom = 1f
    private var xCam = 0f
    private var yCam = 0f

    private lateinit var sim: Simulator

    override fun settings() {
        size(SCREEN_WIDTH, SCREEN_HEIGHT)
        noSmooth()
    }

    override fun setup() {
        frameRate(165f/4f)

        loadPixels()
        updatePixels()

        sim = Simulator(WORLD_WIDTH, WORLD_HEIGHT, this, (8192 * 1024) / 16, "main_cl_program.cl")
    }

    override fun draw() {
//        if (noDraw)
//            for (i in 0 until 1000)
//                sim.run()
//        else {
//            sim.run()
//            if (upPressed) yCam -= max(8/zoom, 1f)
//            if (downPressed) yCam += max(8/zoom, 1f)
//            if (leftPressed) xCam -= max(8/zoom, 1f)
//            if (rightPressed) xCam += max(8/zoom, 1f)
            if (upPressed) yCam -= 8/zoom
            if (downPressed) yCam += 8/zoom
            if (leftPressed) xCam -= 8/zoom
            if (rightPressed) xCam += 8/zoom
//        }

        if (frameCount % 10 == 0)
            sim.run()


        loadPixels()
        sim.render(xCam, yCam, zoom, (frameCount % 10) / 10f)
        updatePixels()

        textAlign(LEFT, TOP)
        text("FPS: $frameRate", 0f, 0f)


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