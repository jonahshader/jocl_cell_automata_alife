package jonahshader

import processing.core.PApplet
import processing.core.PConstants
import processing.event.KeyEvent

class App : PApplet() {
    companion object {
        const val SCREEN_WIDTH = 1920
        const val SCREEN_HEIGHT = 1080

        const val WORLD_WIDTH = 1920
        const val WORLD_HEIGHT = 1080
    }

    private val noDrawToggleKey = 'o'
    private var noDrawKeyPressed = false
    private var noDraw = false

    private val sim = Simulator(WORLD_WIDTH, WORLD_HEIGHT, 100000, "main_cl_program.cl")

    override fun settings() {
        size(SCREEN_WIDTH, SCREEN_HEIGHT)
        noSmooth()
    }

    override fun setup() {
        frameRate(165f)
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
        for (i in 0 until SCREEN_WIDTH * SCREEN_HEIGHT) {
//            stroke(color(if (world[i].toInt() == 0) 0 else 255))
//            point((i % SCREEN_WIDTH).toFloat(), (i / SCREEN_HEIGHT).toFloat())
            pixels[i] = color(if (world[i].toInt() == -1) 0 else 255)
        }

        updatePixels()

        textAlign(LEFT, TOP)
        text("FPS: $frameRate", 0f, 0f)


    }

    override fun keyPressed() {
//        when (key.toLowerCase()) {
//            noDrawToggleKey -> {
//                noDrawKeyPressed = true
//                noDraw = !noDraw
//            }
//        }
    }

    override fun keyPressed(event: KeyEvent?) {
        when (event?.key?.toLowerCase()) {
            noDrawToggleKey -> {
                noDrawKeyPressed = true
                noDraw = !noDraw
                println("fuck")
            }
        }
    }

    override fun keyReleased() {
        when (key.toLowerCase()) {
            noDrawToggleKey -> {noDrawKeyPressed = false
            println("fuck")}
        }
    }
}