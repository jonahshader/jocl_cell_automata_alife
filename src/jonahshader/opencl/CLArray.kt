package jonahshader.opencl

import org.jocl.cl_kernel

interface CLArray {
    fun registerAndSendArgument(kernel: cl_kernel?, argIndex: Int)
    fun copyToDevice()
    fun copyFromDevice()
    fun dispose()
}