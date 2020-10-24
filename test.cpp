#include <iostream>
#include <cstring>
int main() {
	float a[3] = {1,2,3};
	void *data;
	void *data2 = (float*)malloc(2*sizeof(float));
	data = (float*)malloc(3*sizeof(float));
	memcpy(data, a, 3*sizeof(float));
	memcpy(data2, data+sizeof(float), 2*sizeof(float));
	std::cout << a[0] << std::endl;
	std::cout << *((float*)data2)  << std::endl;
	//std::cout << (float*)(data+sizeof(float)) << std::endl;
	return 0;
}
