#include <boost/python.hpp>
#include "bloom_filter.hpp"
using namespace boost::python;

BOOST_PYTHON_MODULE(google_bloom) {
    class_<bloom_parameters>("BloomParameters")
        .def("compute_optimal_paramters", &bloom_parameters::compute_optimal_paramters);
        .def_readwrite("projected_element_count", &bloom_parameters::projected_element_count)
        .def_readwrite("false_positive_probability", &bloom_parameters::false_positive_probability)
        .def_readwrite("random_seed", &bloom_parameters::random_seed);

    class_<bloom_parameters::optimal_parameters_t>("OptimalParameters");

    class_<bloom_filter>("BloomFilter")
        .def(init<bloom_parameters>())
        .def(init<bloom_filter>())
        .def("clear", &bloom_filter::clear)
        .def("insert", &bloom_filter::insert0)
        .def("insert", &bloom_filter::insert1)
        .def("insert", &bloom_filter::insert2)
        .def("insert", &bloom_filter::insert3)
        .def("insert", &bloom_filter::insert4)
        .def("contains", &bloom_filter::contains0)
        .def("contains", &bloom_filter::contains1)
        .def("contains", &bloom_filter::contains2)
        .def("contains", &bloom_filter::contains3)
        .def("contains", &bloom_filter::contains4)
        .def("contains_all", &bloom_filter::contains_all)
        .def("contains_none", &bloom_filter::contains_none)
        .def("size", &bloom_filter::size)
        .def("element_count", &bloom_filter::element_count)
        .def("effective_fpp", &bloom_filter::effective_fpp);
}

