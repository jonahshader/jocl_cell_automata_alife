package jonahshader.opencl;

import org.jocl.Pointer;
import org.jocl.cl_command_queue;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.stream.Stream;

import static org.jocl.CL.CL_DEVICE_TYPE_ALL;

public class OpenCLSetup {
    private ArrayList<int[]> intArrays = new ArrayList<>();
    private ArrayList<float[]> floatArrays = new ArrayList<>();
    private ArrayList<Pointer> pointers = new ArrayList<>();
    private ArrayList<cl_mem> memory = new ArrayList<>();


    private final int platformIndex = 0;
    private final long deviceType = CL_DEVICE_TYPE_ALL;
    private final int deviceIndex = 0;

    private cl_command_queue commandQueue;
    private cl_kernel kernel;
    private long[] global_work_size;

    private static String programSource;

    public OpenCLSetup(String filename) {
        StringBuilder stringBuilder = new StringBuilder();
        try (Stream<String> stream = Files.lines(Paths.get(filename), StandardCharsets.UTF_8)) {
            stream.forEach(s -> stringBuilder.append(s).append("\n"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        programSource = stringBuilder.toString();

    }



}
