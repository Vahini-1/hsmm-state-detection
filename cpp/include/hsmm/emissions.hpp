#pragma once

#include "hsmm/types.hpp"

namespace hsmm {

// log Student-t density: log P(y | mu, sigma, nu)
double log_student_t_pdf(double y, const EmissionParams& params);

// Vectorized version: log P(y_t | state k) for all t, given fixed params.
Vector log_emission_density(const Vector& observations, const EmissionParams& params);

// M-step update: re-estimate (mu, sigma, nu) for a regime given
// responsibility-weighted observations (EM weighted MLE, e.g. via a
// few Newton/EM-within-EM iterations for the Student-t nu parameter).
EmissionParams fit_emission_params(const Vector& observations,
                                    const Vector& responsibilities);

}  // namespace hsmm
