#include "core/plant.hpp"


namespace core {

int PendulumParams::n_links() const {
    // TODO: masses.size().
    return static_cast<int>(masses.size());
}

PendulumParams PendulumParams::uniform_rods(int n_links, double mass, double length,
                                            double damping, double carriage_mass) {
    // Bounds-checked before the vectors are sized, not after: Vec has fixed inline
    // storage for kMaxLinks, so an over-length Constant() overruns it, and the
    // eigen_assert that would catch that is compiled out in the release builds
    // where training actually runs.
    if (n_links < 1 || n_links > kMaxLinks) {
        throw std::invalid_argument(
            std::format("n_links = {} must be in [1, {}]", n_links, kMaxLinks));
    }

    auto com_dist = length/2.0;
    auto inertia = mass * (length * length)/12.0;
    PendulumParams ret_val{
        .carriage_mass = carriage_mass,
        .masses = Vec::Constant(n_links, mass),
        .lengths = Vec::Constant(n_links, length),
        .com_dists = Vec::Constant(n_links, com_dist),
        .inertias = Vec::Constant(n_links, inertia),
        .joint_damping = Vec::Constant(n_links, damping)};
    ret_val.validate();
    return ret_val;
}

void PendulumParams::validate() const {
    // TODO: throw if the five per-link vectors disagree in length, if n_links is
    // outside [1, kMaxLinks], or if any mass/length/inertia is non-positive.
    // Cheap, and it converts a domain-randomisation bug from NaNs deep in a
    // rollout into an error at the point of construction.
    const int n = n_links();
    if (!masses.allFinite() || !lengths.allFinite() || !com_dists.allFinite() ||
        !inertias.allFinite() || !joint_damping.allFinite()) {
        throw std::invalid_argument("params contain NaN or inf");
    }   
    else if (!(n >= 1 && n <= kMaxLinks)){
        throw std::invalid_argument(std::format("Number of links must be between 1 and {}", kMaxLinks));
    }
    else if (lengths.size() != n || com_dists.size() != n ||
        inertias.size() != n || joint_damping.size() != n) {
        throw std::invalid_argument("per-link vectors disagree in length");
    }
    else if (!(masses.array() > 0.0).all()){
        Eigen::Index i;
        const double lo = masses.minCoeff(&i);
        throw std::invalid_argument(std::format("masses[{}] = {} must be positive", i, lo));
    }
    else if (!(lengths.array() > 0.0).all()){
        Eigen::Index i;
        const double lo = lengths.minCoeff(&i);
        throw std::invalid_argument(std::format("lengths[{}] = {} must be positive", i, lo));
    }
    else if (!(inertias.array() > 0.0).all()){
        Eigen::Index i;
        const double lo = inertias.minCoeff(&i);
        throw std::invalid_argument(std::format("inertias[{}] = {} must be positive", i, lo));
    }
    else if (!(joint_damping.array() >= 0.0).all()){
        Eigen::Index i;
        const double lo = joint_damping.minCoeff(&i);
        throw std::invalid_argument(std::format("joint_damping[{}] = {} must be >= 0.0", i, lo));
    }
    else if (!(carriage_mass > 0.0)) {
        throw std::invalid_argument(std::format("carriage_mass = {} must be positive", carriage_mass));
    }
    else if (!(com_dists.array() > 0.0).all()) {
        Eigen::Index i;
        const double lo = com_dists.minCoeff(&i);
        throw std::invalid_argument(std::format("com_dists[{}] = {} must be positive", i, lo));
    }
    else if (!(com_dists.array() <= lengths.array()).all()) {
        Eigen::Index i;
        (com_dists.array() / lengths.array()).maxCoeff(&i);   // worst offender
        throw std::invalid_argument(std::format(
            "com_dists[{}] = {} exceeds lengths[{}] = {}", i, com_dists[i], i, lengths[i]));
    }
}

Plant::Plant(const PendulumParams& params, double gravity): gravity_(gravity){
    params.validate();
    params_ = params;
    
    n_ = params.n_links();
    build_inertia_terms();
}

int Plant::n_links() const {
    return this->params().n_links();
}

const PendulumParams& Plant::params() const {
    return params_;
}

void Plant::build_inertia_terms() {
    h_.resize(n_);
    D_.resize(n_, n_);
    double mass_beyond_i = 0.0; //First pass is last link only. i = n_, no mass past last link, hence the  0.0
    for(int i = n_ - 1; i >= 0; --i){
        h_[i] = params_.masses[i]*params_.com_dists[i]+params_.lengths[i]*mass_beyond_i;
        D_(i,i) = params_.inertias[i] + params_.masses[i]*params_.com_dists[i]*params_.com_dists[i] + params_.lengths[i]*params_.lengths[i]*mass_beyond_i;
        mass_beyond_i += params_.masses[i]; //First loop is past 2nd last link. Only mass is last link, which is i = n_ - 1. 3 links means i = 2 for mass of last link
    }

    for(int i = 0; i < n_; ++i){
        for (int k = i + 1; k < n_; ++k) {
            D_(i, k) = params_.lengths[i] * h_[k];
            D_(k, i) = D_(i, k);
        }
    }

    total_mass_ = params_.carriage_mass + params_.masses.sum();
    
}

Vec Plant::damping_torques(const Vec& omega) const {
    //   tau_i = b_i (om_i - om_{i-1}),  om_{-1} = 0
    //   Q_i   = -tau_i + tau_{i+1},     tau_n  = 0
    // Each joint torque appears in two equations with opposite sign -- dropping the
    // reaction term is a common slip and shows up as energy drift in the test.
    Vec Q(n_), tau(n_);
    for (int i = 0; i < n_; ++i) {
        tau[i] = params_.joint_damping[i] * (i == 0 ? omega[0] : omega[i] - omega[i-1]);
    }
    for (int i = 0; i < n_; ++i) {
        Q[i] = -tau[i] + (i + 1 < n_ ? tau[i+1] : 0.0);
    }
    return Q;
}

Derivatives Plant::accel(const Vec& theta, const Vec& omega, double carriage_accel) const {
    // TODO: one dynamics evaluation.
    //
    //   A(th) alpha = g (h .* sin th) - a (h .* cos th) - C(th) (om .* om) + Q
    //   A_ik = D_ik cos(th_i - th_k),   C_ik = D_ik sin(th_i - th_k)
    //
    // Evaluate sin/cos once per link and expand the differences:
    //   cos(a-b) = cos a cos b + sin a sin b
    //   sin(a-b) = sin a cos b - cos a sin b
    // At N=3 that is 6 transcendentals instead of 18, and they dominate this call.
    //
    // A is a mass matrix -- symmetric positive definite -- so solve with LDLT.
    //
    // Then the belt force, from the carriage row of the Lagrangian:
    //   F = total_mass_ * a + sum_j h_j (cos th_j alpha_j - sin th_j om_j^2)
    
    Derivatives ret_val;
    Mat A(n_, n_);
    Mat C(n_, n_);

    Vec s = theta.array().sin();
    Vec c = theta.array().cos();


    for(int i = 0; i < n_; ++i){
        A(i, i) = D_(i, i);   // cos(0) = 1
        C(i, i) = 0.0;        // sin(0) = 0
        for (int k = i + 1; k < n_; ++k) {
            // cos is even, so A is symmetric; sin is odd, so C is antisymmetric.
            A(i, k) = A(k, i) = D_(i, k)*(c[i]*c[k] + s[i]*s[k]);
            C(i, k) = D_(i, k)*(s[i]*c[k] - c[i]*s[k]);
            C(k, i) = -C(i, k);
        }
    }

    Vec b;
    b = (gravity_ * h_.array() * s.array()).matrix()
        - (carriage_accel*h_.array()*c.array()).matrix()
        - C * omega.cwiseAbs2()
        + damping_torques(omega);


    ret_val.alpha = A.ldlt().solve(b);

    ret_val.belt_force = total_mass_ * carriage_accel
            + (h_.array()*(c.array()*ret_val.alpha.array() - s.array()*omega.array().square())).sum();

    return ret_val;
}

void Plant::step(State& state, double carriage_accel, double dt,
                 double* peak_belt_force) const {
    // RK4 over (theta, omega) at zero-order-hold carriage_accel, advancing
    // state in place. The carriage is prescribed, so x and xdot integrate directly
    // from carriage_accel rather than coming out of the solve.
    //
    // If peak_belt_force is non-null, write max |force| across the four stages --
    // the step-loss check needs the peak, not the value at any single stage.

    auto k1 = accel(state.theta, state.omega, carriage_accel);

    auto k2 = accel(state.theta + (dt/2)*state.omega,
                    state.omega + (dt/2)*k1.alpha, 
                    carriage_accel);

    auto k3 = accel(state.theta + (dt/2)*(state.omega + (dt/2)*k1.alpha),
                    state.omega + (dt/2)*k2.alpha, 
                    carriage_accel);

    auto k4 = accel(state.theta + dt*(state.omega + (dt/2)*k2.alpha),
                    state.omega + dt*k3.alpha, 
                    carriage_accel);

    state.theta += (dt/6)*(6*state.omega + dt*k1.alpha + dt*k2.alpha + dt*k3.alpha);
    state.omega += (dt/6)*(k1.alpha + 2*k2.alpha + 2*k3.alpha + k4.alpha);
    state.x += state.xdot * dt + 0.5*carriage_accel*dt*dt;
    state.xdot += carriage_accel*dt;

    if (peak_belt_force != nullptr) {
        // Nested two-argument std::max rather than the initializer_list overload:
        // that one routes through the STL's vectorised max_element, which is a
        // separately-compiled library symbol (__std_max_element_d) and so ties this
        // object to a matching runtime. core has to link against whatever toolchain
        // the Pi has; the two-argument form is a header-only template.
        const double a = std::max(std::abs(k1.belt_force), std::abs(k2.belt_force));
        const double b = std::max(std::abs(k3.belt_force), std::abs(k4.belt_force));
        *peak_belt_force = std::max(a, b);
    }
}

double Plant::energy(const Vec& theta, const Vec& omega, double xdot) const {
    //   T = 0.5 total_mass_ xdot^2
    //       + xdot sum_j h_j cos th_j om_j
    //       + 0.5 sum_jk D_jk cos(th_j - th_k) om_j om_k
    //   V = g sum_j h_j cos th_j
    //
    // This is the validation hook: with zero damping and zero carriage acceleration
    // T + V must hold constant. Test at N=1, 2 and 3 -- a sign error in the D_ik
    // cross terms is entirely invisible at N=1.
    Vec s = theta.array().sin();
    Vec c = theta.array().cos();
    
    double T = 0.5*total_mass_*xdot*xdot
           + xdot*(h_.array()*c.array()*omega.array()).sum();
           

    for(int j = 0; j < n_; ++j){
        T += 0.5*D_(j, j)*omega[j]*omega[j];
        for (int k = j + 1; k < n_; ++k) {
            T += D_(j, k)*(c[j]*c[k] + s[j]*s[k])*omega[j]*omega[k];
        }
    }

    double V = gravity_*(h_.array()*c.array()).sum();

    return T + V;
}

}  // namespace core
