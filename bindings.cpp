#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "hsmm/types.hpp"
#include "hsmm/forward_backward.hpp"
#include "hsmm/em_engine.hpp"
#include "hsmm/particle_filter.hpp"
#include "hsmm/emissions.hpp"
#include "hsmm/duration_distributions.hpp"

namespace py = pybind11;
using namespace hsmm;

namespace {

// --- Conversion helpers between our lightweight Matrix/Vector types and
//     numpy arrays, so the Python side works with ndarray in/out rather
//     than opaque C++ objects. ---

py::array_t<double> matrix_to_numpy(const Matrix& m) {
    py::array_t<double> arr({m.rows, m.cols});
    auto buf = arr.mutable_unchecked<2>();
    for (std::size_t i = 0; i < m.rows; ++i) {
        for (std::size_t j = 0; j < m.cols; ++j) {
            buf(i, j) = m(i, j);
        }
    }
    return arr;
}

Matrix numpy_to_matrix(const py::array_t<double>& arr) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument("expected a 2-D array for Matrix conversion");
    }
    auto buf = arr.unchecked<2>();
    Matrix m(static_cast<std::size_t>(arr.shape(0)), static_cast<std::size_t>(arr.shape(1)));
    for (py::ssize_t i = 0; i < arr.shape(0); ++i) {
        for (py::ssize_t j = 0; j < arr.shape(1); ++j) {
            m(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = buf(i, j);
        }
    }
    return m;
}

Vector numpy_to_vector(const py::array_t<double>& arr) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument("expected a 1-D array for Vector conversion");
    }
    auto buf = arr.unchecked<1>();
    Vector v(static_cast<std::size_t>(arr.shape(0)));
    for (py::ssize_t i = 0; i < arr.shape(0); ++i) v[static_cast<std::size_t>(i)] = buf(i);
    return v;
}

py::array_t<double> vector_to_numpy(const Vector& v) {
    py::array_t<double> arr(v.size());
    auto buf = arr.mutable_unchecked<1>();
    for (std::size_t i = 0; i < v.size(); ++i) buf(i) = v[i];
    return arr;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ engine for Bayesian HSMM regime modeling (forward-backward, "
              "EM, and online particle filter changepoint detection).";

    // ---------------------------------------------------------------
    // Parameter structs
    // ---------------------------------------------------------------
    py::class_<EmissionParams>(m, "EmissionParams")
        .def(py::init<>())
        .def(py::init([](double mu, double sigma, double nu) {
                 EmissionParams p;
                 p.mu = mu;
                 p.sigma = sigma;
                 p.nu = nu;
                 return p;
             }),
             py::arg("mu") = 0.0, py::arg("sigma") = 1.0, py::arg("nu") = 5.0)
        .def_readwrite("mu", &EmissionParams::mu)
        .def_readwrite("sigma", &EmissionParams::sigma)
        .def_readwrite("nu", &EmissionParams::nu)
        .def("__repr__", [](const EmissionParams& p) {
            return "EmissionParams(mu=" + std::to_string(p.mu) +
                   ", sigma=" + std::to_string(p.sigma) + ", nu=" + std::to_string(p.nu) + ")";
        });

    py::class_<DurationParams>(m, "DurationParams")
        .def(py::init<>())
        .def(py::init([](double r, double p, std::size_t max_duration) {
                 DurationParams d;
                 d.r = r;
                 d.p = p;
                 d.max_duration = max_duration;
                 return d;
             }),
             py::arg("r") = 1.0, py::arg("p") = 0.5, py::arg("max_duration") = 252)
        .def_readwrite("r", &DurationParams::r)
        .def_readwrite("p", &DurationParams::p)
        .def_readwrite("max_duration", &DurationParams::max_duration)
        .def("__repr__", [](const DurationParams& d) {
            return "DurationParams(r=" + std::to_string(d.r) + ", p=" + std::to_string(d.p) +
                   ", max_duration=" + std::to_string(d.max_duration) + ")";
        });

    py::class_<HSMMParams>(m, "HSMMParams")
        .def(py::init<>())
        .def_readwrite("n_states", &HSMMParams::n_states)
        .def_property(
            "transition",
            [](const HSMMParams& p) { return matrix_to_numpy(p.transition); },
            [](HSMMParams& p, const py::array_t<double>& arr) { p.transition = numpy_to_matrix(arr); })
        .def_readwrite("emissions", &HSMMParams::emissions)
        .def_readwrite("durations", &HSMMParams::durations)
        .def_property(
            "initial_dist",
            [](const HSMMParams& p) { return vector_to_numpy(p.initial_dist); },
            [](HSMMParams& p, const py::array_t<double>& arr) { p.initial_dist = numpy_to_vector(arr); })
        .def("__repr__", [](const HSMMParams& p) {
            return "HSMMParams(n_states=" + std::to_string(p.n_states) + ")";
        });

    // ---------------------------------------------------------------
    // Forward-backward
    // ---------------------------------------------------------------
    py::class_<ForwardBackwardResult>(m, "ForwardBackwardResult")
        .def_property_readonly("log_alpha",
                                [](const ForwardBackwardResult& r) { return matrix_to_numpy(r.log_alpha); })
        .def_property_readonly("log_beta",
                                [](const ForwardBackwardResult& r) { return matrix_to_numpy(r.log_beta); })
        .def_property_readonly("log_gamma",
                                [](const ForwardBackwardResult& r) { return matrix_to_numpy(r.log_gamma); })
        .def_readonly("log_likelihood", &ForwardBackwardResult::log_likelihood);

    m.def(
        "forward_backward",
        [](const py::array_t<double>& observations, const HSMMParams& params) {
            return forward_backward(numpy_to_vector(observations), params);
        },
        py::arg("observations"), py::arg("params"),
        "Run the explicit-duration HSMM forward-backward algorithm.");

    // ---------------------------------------------------------------
    // EM engine
    // ---------------------------------------------------------------
    py::class_<EMConfig>(m, "EMConfig")
        .def(py::init<>())
        .def_readwrite("max_iters", &EMConfig::max_iters)
        .def_readwrite("tol", &EMConfig::tol)
        .def_readwrite("seed", &EMConfig::seed)
        .def_readwrite("verbose", &EMConfig::verbose);

    py::class_<EMResult>(m, "EMResult")
        .def_readonly("params", &EMResult::params)
        .def_readonly("log_likelihood_history", &EMResult::log_likelihood_history)
        .def_readonly("converged", &EMResult::converged);

    m.def(
        "fit_hsmm_em",
        [](const py::array_t<double>& observations, const HSMMParams& init_params,
           const EMConfig& config) {
            return fit_hsmm_em(numpy_to_vector(observations), init_params, config);
        },
        py::arg("observations"), py::arg("init_params"), py::arg("config") = EMConfig{},
        "Fit HSMM parameters via Baum-Welch-style EM with explicit durations.");

    m.def(
        "most_likely_state_path",
        [](const py::array_t<double>& observations, const HSMMParams& params) {
            auto path = most_likely_state_path(numpy_to_vector(observations), params);
            py::array_t<std::size_t> arr(path.size());
            auto buf = arr.mutable_unchecked<1>();
            for (std::size_t i = 0; i < path.size(); ++i) buf(i) = path[i];
            return arr;
        },
        py::arg("observations"), py::arg("params"),
        "Segmental Viterbi decoding: most likely regime path given fitted params.");

    // ---------------------------------------------------------------
    // Particle filter (stateful class, used for streaming/online use)
    // ---------------------------------------------------------------
    py::class_<ParticleFilterConfig>(m, "ParticleFilterConfig")
        .def(py::init<>())
        .def_readwrite("n_particles", &ParticleFilterConfig::n_particles)
        .def_readwrite("ess_resample_threshold", &ParticleFilterConfig::ess_resample_threshold)
        .def_readwrite("seed", &ParticleFilterConfig::seed);

    py::class_<Particle>(m, "Particle")
        .def_readonly("state", &Particle::state)
        .def_readonly("time_in_state", &Particle::time_in_state)
        .def_readonly("log_weight", &Particle::log_weight);

    py::class_<ParticleFilter>(m, "ParticleFilter")
        .def(py::init<const HSMMParams&, const ParticleFilterConfig&>(), py::arg("params"),
             py::arg("config") = ParticleFilterConfig{})
        .def(
            "step",
            [](ParticleFilter& pf, double y_t) { return vector_to_numpy(pf.step(y_t)); },
            py::arg("y_t"),
            "Advance the filter by one observation; returns P(z_t = k | y_1:t) as an array.")
        .def("effective_sample_size", &ParticleFilter::effective_sample_size)
        .def("particles", &ParticleFilter::particles,
             "Return the current particle population (for diagnostics).");

    // ---------------------------------------------------------------
    // Standalone emission / duration helpers (useful for diagnostics,
    // posterior predictive checks, and unit tests from the Python side).
    // ---------------------------------------------------------------
    m.def("log_student_t_pdf", &log_student_t_pdf, py::arg("y"), py::arg("params"));
    m.def(
        "log_emission_density",
        [](const py::array_t<double>& observations, const EmissionParams& params) {
            return vector_to_numpy(log_emission_density(numpy_to_vector(observations), params));
        },
        py::arg("observations"), py::arg("params"));

    m.def("log_duration_pmf", &log_duration_pmf, py::arg("d"), py::arg("params"));
    m.def("log_duration_survivor", &log_duration_survivor, py::arg("d"), py::arg("params"));
    m.def(
        "precompute_log_duration_pmf",
        [](const DurationParams& params) { return vector_to_numpy(precompute_log_duration_pmf(params)); },
        py::arg("params"));
}
