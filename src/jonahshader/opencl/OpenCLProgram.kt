package jonahshader.opencl

import org.jocl.*
import java.io.IOException
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Paths
import java.util.*

/**
 * this OpenCLProgram class takes in the filename of a .cl OpenCL program and a list of kernel names
 * that are found in the program. it then generates the corresponding kernels. you can run a kernel by
 * calling executeKernel and providing the name of the kernel.
 *
 * this class should only be created once per session.
 */
class OpenCLProgram(filename: String, kernelNames: Array<String>) {
    private val platformIndex = 0
    private val deviceType = CL.CL_DEVICE_TYPE_ALL
    private val deviceIndex = 0

    lateinit var commandQueue: cl_command_queue
    private val kernels = HashMap<String, cl_kernel>()
    private val programSource: String
    private lateinit var program: cl_program
    lateinit var context: cl_context

    init { // get source code from file
        val stringBuilder = StringBuilder()
        Files.lines(Paths.get(filename), StandardCharsets.UTF_8).use { stream -> stream.forEach { s: String? -> stringBuilder.append(s).append("\n") } }
        programSource = stringBuilder.toString()
        // do a bunch of opencl setup stuff
        clInit()
        // create kernels from opencl program
        for (i in kernelNames.indices) {
            kernels[kernelNames[i]] = CL.clCreateKernel(program, kernelNames[i], null)
        }
    }

    fun getKernel(kernelName: String): cl_kernel? {
        return kernels[kernelName]
    }

    fun executeKernel(kernelName: String, range: Long) {
        CL.clEnqueueNDRangeKernel(commandQueue, kernels[kernelName]!!,
                1, null, longArrayOf(range),
                null, 0, null, null)
    }

    fun waitForCL() {
        CL.clFinish(commandQueue)
    }

    fun createCLIntArray(size: Int) : CLIntArray = CLIntArray(IntArray(size), context, commandQueue)

    fun createCLCharArray(size: Int) : CLCharArray = CLCharArray(ByteArray(size), context, commandQueue)

    fun createCLShortArray(size: Int) : CLShortArray = CLShortArray(ShortArray(size), context, commandQueue)

    fun createCLFloatArray(size: Int) : CLFloatArray = CLFloatArray(FloatArray(size), context, commandQueue)

    // opencl setup stuff
    private fun clInit() {
        CL.setExceptionsEnabled(true)

        // Obtain the number of platforms
        val numPlatformsArray = IntArray(1)
        CL.clGetPlatformIDs(0, null, numPlatformsArray)
        val numPlatforms = numPlatformsArray[0]

        // Obtain a platform ID
        val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
        CL.clGetPlatformIDs(platforms.size, platforms, null)
        val platform = platforms[platformIndex]

        // Initialize the context properties
        val contextProperties = cl_context_properties()
        contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM.toLong(), platform)

        // Obtain the number of devices for the platform
        val numDevicesArray = IntArray(1)
        CL.clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray)
        val numDevices = numDevicesArray[0]

        // Obtain a device ID
        val devices = arrayOfNulls<cl_device_id>(numDevices)
        CL.clGetDeviceIDs(platform, deviceType, numDevices, devices, null)
        val device = devices[deviceIndex]

        // Create a context for the selected device
        context = CL.clCreateContext(
                contextProperties, 1, arrayOf(device),
                null, null, null)

        // Create a command-queue for the selected device
        val properties = cl_queue_properties()
        commandQueue = CL.clCreateCommandQueueWithProperties(
                context, device, properties, null)

        // Create the program from the source code
        program = CL.clCreateProgramWithSource(context,
                1, arrayOf(programSource), null, null)

        // Build the program
        CL.clBuildProgram(program, 0, null, null, null, null)
    }
}