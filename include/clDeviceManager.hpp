#ifndef CLDEVICEMANAGER_HPP
#define CLDEVICEMANAGER_HPP

#include <fstream>
#include <string>

#include <clBLAS.h>

enum PRG{
	CLMAT_EYE = 0,
	CLMAT_ONES,
	CLMAT_ZEROS,
	CLMAT_HADAMARD,
	CLMAT_SUB,
	LENG
};

const static std::string PRG_NAME[] = {
	"clMatrix_eye",
	"clMatrix_ones",
	"clMatrix_zeros",
	"clMatrix_hadamard",
	"clMatrix_sub"
};

class clDeviceManager
{
private:
	cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context ctx = 0;
    cl_command_queue queue = 0;

	cl_kernel kernel[PRG::LENG];
	
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

	std::string read_program ( const std::string& filename );
public:
	clDeviceManager();
	~clDeviceManager();

	cl_context get_context ();	
	cl_command_queue get_queue ();

	cl_command_queue* get_queue_ptr ();

	void run_kernel ( int kernel_idx, size_t gl_work_size, size_t lc_work_size );
	void set_argument ( int kernel_idx, int arg_idx, void* val );
}cl_device_manager;

std::string clDeviceManager::read_program ( const std::string& filename )
{
	const std::string FILE_HEADER = "../include";
	std::string fn = FILE_HEADER + "/CL/" + filename + ".cl";
	std::ifstream ifs(fn);
	std::string ret = "";

	while( !ifs.eof() ){
		std::string buf;
		getline(ifs, buf);
		printf("%s\n", buf.c_str());
		ret += buf;
	}

	return ret;
}

clDeviceManager::clDeviceManager()
{
	cl_int err;

	err = clGetPlatformIDs( 1, &platform, NULL );
	err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

	props[1] = (cl_context_properties)platform;
	ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
	queue = clCreateCommandQueue( ctx, device, 0, &err );

    err = clblasSetup();

	for( int i = 0; i < PRG::LENG; ++i ){
		std::string source = read_program(PRG_NAME[i]);
		const char* c_source = source.data();
		size_t source_size = source.size();
		cl_int err;
		
		cl_program program = clCreateProgramWithSource(ctx, 1, &c_source, &source_size, &err);
		err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
		printf("%d : build %d\n", i, err);
		kernel[i] = clCreateKernel(program, PRG_NAME[i].c_str(), &err);
		printf("%d : create kernel %d\n", i, err);
	}
}

clDeviceManager::~clDeviceManager()
{
    clblasTeardown( );
    clReleaseCommandQueue( queue );
    clReleaseContext( ctx );
}

cl_context clDeviceManager::get_context ()
{
	return ctx;
}

cl_command_queue clDeviceManager::get_queue ()
{
	return queue;
}

cl_command_queue* clDeviceManager::get_queue_ptr ()
{
	return &queue;
}

void clDeviceManager::run_kernel ( int kernel_idx, size_t gl_work_size, size_t lc_work_size )
{
	cl_event event;

	size_t global_work_size[3] = { gl_work_size, lc_work_size, 0 },
		local_work_size[3] = { 1, 1, 0 };
	
	cl_int err = clEnqueueNDRangeKernel(queue, kernel[kernel_idx], 2, NULL, global_work_size, NULL, 0, NULL, &event);
	clWaitForEvents(1, &event);
}

void clDeviceManager::set_argument ( int kernel_idx, int arg_idx, void* val )
{
	cl_int err = clSetKernelArg(kernel[kernel_idx], arg_idx, sizeof(cl_mem), val);
}

#endif
