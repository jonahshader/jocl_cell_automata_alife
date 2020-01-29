package jonahshader.opencl

import org.jocl.*

class CLIntArray(private val array: IntArray, context: cl_context, private val commandQueue: cl_command_queue) {
    private val memory: cl_mem
    private val hostMemPointer: Pointer = Pointer.to(array)
    private val deviceMemPointer: Pointer

    init {
        memory = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE or CL.CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int * array.size.toLong(), hostMemPointer, null)
        deviceMemPointer = Pointer.to(memory)
    }

    /**
     * sets this array as an argument to a kernel
     * @param kernel - the kernel
     * @param argIndex - the index of the argument
     */
    fun registerAndSendArgument(kernel: cl_kernel?, argIndex: Int) {
        CL.clSetKernelArg(kernel, argIndex, Sizeof.cl_mem.toLong(), deviceMemPointer)
    }

    /**
     * copies this array from host to device
     */
    fun copyToDevice() {
        CL.clEnqueueWriteBuffer(commandQueue, memory, true, 0,
                Sizeof.cl_int * array.size.toLong(), hostMemPointer,
                0, null, null)
    }

    /**
     * copies this array from device to host
     */
    fun copyFromDevice() {
        CL.clEnqueueReadBuffer(commandQueue, memory, true, 0,
                Sizeof.cl_int * array.size.toLong(), hostMemPointer,
                0, null, null)
    }
}