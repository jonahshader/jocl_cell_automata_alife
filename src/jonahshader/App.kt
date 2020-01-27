package jonahshader

import processing.core.PApplet

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
        do {
            // compute
        } while (noDraw)
        // draw
    }

    override fun keyPressed() {
        when (key.toLowerCase()) {
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