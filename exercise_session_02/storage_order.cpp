#include "blitz/array.h"
using namespace blitz;
int main() {
	Range dim1(10,14);
	Range dim2(0,19);
	Range dim3(0,19);	
	GeneralArrayStorage<3> storage;
	Array<int,3> A(dim1,dim2,dim3,storage);
	std::cout << "Original indexes: " << A.lbound() << " - " << A.ubound() << std::endl;
	return 0;
}
	
