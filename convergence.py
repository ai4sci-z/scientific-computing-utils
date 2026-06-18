"""
Convergence analysis utilities for numerical PDE solvers.

Typical usage:
    from convergence import l2_error, convergence_table, plot_convergence
"""

import numpy as np
import matplotlib.pyplot as plt


def l1_error(u_num, u_exact, dx):
    return dx * np.sum(np.abs(u_num - u_exact))


def l2_error(u_num, u_exact, dx):
    return np.sqrt(dx * np.sum((u_num - u_exact)**2))


def linf_error(u_num, u_exact):
    return np.max(np.abs(u_num - u_exact))


def convergence_rate(errors, Ns):
    """Estimate order of convergence from a sequence of (N, error) pairs."""
    rates = []
    for i in range(1, len(errors)):
        r = np.log(errors[i-1] / errors[i]) / np.log(Ns[i] / Ns[i-1])
        rates.append(r)
    return rates


def convergence_table(pairs, label="L2 error"):
    """Print a formatted convergence table.

    pairs : list of (N, error)
    """
    Ns     = [p[0] for p in pairs]
    errors = [p[1] for p in pairs]
    rates  = convergence_rate(errors, Ns)

    header = f"{'N':>6}  {label:>12}  {'order':>7}"
    print(header)
    print("-" * len(header))
    print(f"{Ns[0]:6d}  {errors[0]:12.4e}  {'—':>7}")
    for i in range(1, len(Ns)):
        print(f"{Ns[i]:6d}  {errors[i]:12.4e}  {rates[i-1]:7.2f}")
    return rates


def plot_convergence(pairs_dict, title="Convergence", filename=None):
    """Plot convergence curves for multiple schemes.

    pairs_dict : {'scheme_name': [(N, error), ...], ...}
    """
    styles = ['-o', '-s', '-^', '-D', '-v']
    fig, ax = plt.subplots(figsize=(6, 5))

    for (name, pairs), style in zip(pairs_dict.items(), styles):
        Ns     = np.array([p[0] for p in pairs])
        errors = np.array([p[1] for p in pairs])
        ax.loglog(Ns, errors, style, label=name, lw=1.5, ms=6)

    # reference slopes
    x_ref = np.array([pairs_dict[list(pairs_dict.keys())[0]][0][0],
                       pairs_dict[list(pairs_dict.keys())[0]][-1][0]])
    e0 = list(pairs_dict.values())[0][0][1]
    for order, ls in [(2, '--'), (4, ':'), (5, '-.')]:
        ref = e0 * (x_ref[0] / x_ref)**order
        ax.loglog(x_ref, ref, ls, color='gray', lw=1,
                  label=f'{order}th-order ref')

    ax.set_xlabel('N (grid points)')
    ax.set_ylabel('Error')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Demo: manufactured solution convergence for 1D Poisson FD")

    def solve_poisson(N):
        """Solve -u'' = f on [0,1] with u(0)=u(1)=0.
        Exact: u = sin(pi x), f = pi^2 sin(pi x).
        """
        x  = np.linspace(0, 1, N+2)
        dx = x[1] - x[0]
        xi = x[1:-1]           # interior
        f  = np.pi**2 * np.sin(np.pi * xi)
        A  = (np.diag(2*np.ones(N)) -
              np.diag(np.ones(N-1), 1) -
              np.diag(np.ones(N-1), -1)) / dx**2
        u_num = np.linalg.solve(A, f)
        u_ex  = np.sin(np.pi * xi)
        return l2_error(u_num, u_ex, dx)

    Ns = [10, 20, 40, 80, 160, 320]
    pairs = [(N, solve_poisson(N)) for N in Ns]
    convergence_table(pairs, label="L2 error")
    plot_convergence({'2nd-order FD': pairs}, title='Poisson 1D convergence',
                     filename='poisson_convergence.png')
