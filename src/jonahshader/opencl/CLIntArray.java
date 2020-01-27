package jonahshader.opencl;

import org.jocl.*;

import static org.jocl.CL.*;

public class CLIntArray {
    private int[] array;
    private cl_mem memory;
    private cl_command_queue commandQueue;
    private Pointer hostMemPointer;
    private Pointer deviceMemPointer;

    public CLIntArray(int[] array, cl_context context, cl_command_queue commandQueue) {
        this.array = array;
        this.commandQueue = commandQueue;

        hostMemPointer = Pointer.to(array);
        memory = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int * array.length, hostMemPointer, null);
        deviceMemPointer = Pointer.to(memory);
    }

    /**
     * sets this array as an argument to a kernel
     * @param kernel - the kernel
     * @param argIndex - the index of the argument
     */
    public void registerAndSendArgument(cl_kernel kernel, int argIndex) {
        clSetKernelArg(kernel, argIndex, Sizeof.cl_mem, deviceMemPointer);
    }

    /**
     * copies this array from host to device
     */
    public void send() {
        clEnqueueWriteBuffer(commandQueue, memory, true, 0,
                Sizeof.cl_int * array.length, hostMemPointer,
                0, null, null);
    }

    /**
     * copies this array from device to host
     */
    public void retrieve() {
        clEnqueueReadBuffer(commandQueue, memory, true, 0,
                Sizeof.cl_int * array.length, hostMemPointer,
                0, null, null);
    }
}
