package jonahshader

import processing.core.PApplet
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

    override fun settings() {

    }

    override fun setup() {

    }

    override fun draw() {
        if (noDraw) {
            for (i in 0 until 100) {

            }
        } else {

        }
        background(0)
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