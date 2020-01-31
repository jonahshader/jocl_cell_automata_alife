package jonahshader

import processing.core.PApplet
import processing.core.PConstants
import processing.event.KeyEvent

class App : PApplet() {
    companion object {
        const val SCREEN_WIDTH = 1920
        const val SCREEN_HEIGHT = 1080

        const val WORLD_WIDTH = 8192
        const val WORLD_HEIGHT = 8192
    }

    private val noDrawToggleKey = 'o'
    private var noDrawKeyPressed = false
    private var noDraw = false

    private val sim = Simulator(WORLD_WIDTH, WORLD_HEIGHT, 2073600/40, "main_cl_program.cl")

    override fun settings() {
        size(SCREEN_WIDTH, SCREEN_HEIGHT)
        noSmooth()
    }

    override fun setup() {
        frameRate(60f)
    }

    override fun draw() {
        if (noDraw) {
            for (i in 0 until 100) {
                sim.run()
            }
        } else {
            sim.run()
        }
//        sim.run()

        loadPixels()
        val world = sim.getUpdatedWorld()

        for (y in 0 until SCREEN_HEIGHT) {
            for (x in 0 until SCREEN_WIDTH) {
                pixels[x + y * SCREEN_WIDTH] = color(if (world[x + y * WORLD_WIDTH] == -1) 0 else 255)
            }
        }
//        for (i in 0 until SCREEN_WIDTH * SCREEN_HEIGHT) {
////            if (world[i] == -1) {
////                pixels[i] = color(0)
////            } else {
////                pixels[i] = color((world[i].toInt()) * 255f / 130000.toFloat())
////            }
//
//            pixels[i] = color(if (world[i] == -1) 0 else 255)
//
//        }

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
        }
    }

    override fun keyReleased() {
        when (key.toLowerCase()) {
            noDrawToggleKey -> noDrawKeyPressed = false
        }
    }
}