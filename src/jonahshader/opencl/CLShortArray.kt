package jonahshader.opencl

import org.jocl.*

class CLShortArray(val array: ShortArray, context: cl_context,
                   private val commandQueue: cl_command_queue, private val clp: OpenCLProgram) : CLArray {
    private val memory: cl_mem
    private val hostMemPointer: Pointer = Pointer.to(array)
    private val deviceMemPointer: Pointer

    init {
        memory = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE,
                Sizeof.cl_short * array.size.toLong(), hostMemPointer, null)
        deviceMemPointer = Pointer.to(memory)
        clp.arraysToDispose.add(this)
    }

    /**
     * sets this array as an argument to a kernel
     * @param kernel - the kernel
     * @param argIndex - the index of the argument
     */
    override fun registerAndSendArgument(kernel: cl_kernel?, argIndex: Int) {
        CL.clSetKernelArg(kernel, argIndex, Sizeof.cl_mem.toLong(), deviceMemPointer)
    }

    /**
     * copies this array from host to device
     */
    override fun copyToDevice() {
        CL.clEnqueueWriteBuffer(commandQueue, memory, true, 0,
                Sizeof.cl_short * array.size.toLong(), hostMemPointer,
                0, null, null)
    }

    /**
     * copies this array from device to host
     */
    override fun copyFromDevice() {
        CL.clEnqueueReadBuffer(commandQueue, memory, true, 0,
                Sizeof.cl_short * array.size.toLong(), hostMemPointer,
                0, null, null)
    }

    override fun dispose() {
        CL.clReleaseMemObject(memory)
        clp.arraysToDispose.remove(this)
    }
}