cd $CAFFE_ROOT
./data/mnist/get_mnist.sh
./examples/mnist/create_mnist.sh
./build/tools/caffe train --solver=/path/to/your/caffe_cnn_solver.prototxt
