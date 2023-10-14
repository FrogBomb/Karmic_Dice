from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from scipy import stats
import matplotlib.pyplot as plt

@dataclass
class ArmaGenerator:
    ar_coeffs: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    ma_coeffs: NDArray[np.float64] = field(default_factory=lambda:np.array([1.0]))

    noise_samples: NDArray[np.float64] = field(init=False)
    prev_outputs: NDArray[np.float64] = field(init=False)
    seed_sd: float = 1.0

    def __post_init__(self):
        assert(len(self.ma_coeffs) > 0)
        self.noise_samples = np.random.standard_normal(len(self.ma_coeffs))
        self.prev_outputs = np.random.standard_normal(len(self.ar_coeffs)) * self.seed_sd

    def __next__(self):
        return self.next()
    
    def next(self): 

        next_roll = self.ma_coeffs.dot(self.noise_samples) + self.ar_coeffs.dot(self.prev_outputs)
        self.noise_samples = np.insert(self.noise_samples, 0, np.random.standard_normal())[:-1]
        self.prev_outputs = np.insert(self.prev_outputs, 0, next_roll)[:-1]
        return next_roll

class SimpleArmaRoller():
    """
        Create a roller from a weakly stable ARMA(1, n) or MA(n) process
        
        If e_i are standard normal variables, will generate an arma process of the form
        
        X_t = phi*X_(t-1) + e_t + sum(e_(t-1-i)*theta[i])

        Then, to determine a roll, this will take:
        
        uniform_roll := norm_cdf(X_t, scale=sd)
        
        where

        sd := Standard deviation of X_t

        This is calulated directly from the input parameters.
        
        Note - the rolls will drift from uniform the closer to 1 the absolute value of phi is. 
        Although the very long term effects are uniform, roll-to-roll predictabilty may be highly correlated, and will not seem uniform.

    """
    def __init__(self, phi: float = 0.0, theta: NDArray[np.float64] | float = -1.0):
        assert(abs(phi) < 1) 
        # notes: 
        #   cross_comp := 2*phi*cov(X_(t-1), sum(theta <dot> eps))
        #   cov(X_i * eps_j) := phi*cov(X_i-1 * eps_j) + theta_(i-j) [if i > j]
        #   cov(X_i * eps_i) = theta_0 = 1.0
        def cov_x_eps(i, j):
            if i > j:
                return phi*cov_x_eps(i-1, j) + theta[i - j - 1]
            elif i == j:
                return 1.0
            else:
                return 0.0
            
        cross_comp = 2*phi*theta if isinstance(theta, (int, float)) else 0.0 if phi == 0.0 else 2*phi*sum(t*cov_x_eps(0, -j) for j, t in enumerate(theta))
        self.sd: float = np.sqrt((1 + np.sum(theta**2) + cross_comp)/(1.0 - phi**2))
        ma_coeffs = np.insert(
            theta,
            0,
            1.0,
        )
        self.arma = ArmaGenerator(
            ar_coeffs=np.array([phi]), 
            ma_coeffs=ma_coeffs,
            seed_sd=self.sd
        )

    def roll(self, sides = 20):
        norm_roll = self.arma.next()
        roll = stats.norm.cdf(norm_roll, scale=self.sd)
        return int(roll*sides) + 1

@dataclass
class KarmaRoller:
    """
    Implementations of various karma-based rolling methods utilizing a tracked "karma" metric

    "heat" is a multiplier on the result of the roll from a uniform distribution, and is applied before adding it to the result of the roll.

    "streak" and "max_streak" are just used as metrics for now. 
    """
    karma: float = 0.0
    heat: float = 1.0
    streak: float = 0.0
    max_streak: float = 0.0


    def registerRoll(self, roll_on_unit_interval: float):

        zero_based_roll = roll_on_unit_interval * 2 - 1
        self.karma += zero_based_roll * self.heat

        if (self.streak * zero_based_roll >= 0):
            self.streak += zero_based_roll * self.heat
        else:
            self.streak = 0.0

        if(abs(self.streak) > abs(self.max_streak)):
            self.max_streak = self.streak

    def advantageBasedRoll(self, sides = 20):
        """Perform a roll that rolls `floor(abs(karma))` dice, and takes the least or greatest outcome based on karma
        
        Note - this should have an approximate distribition of Beta(1, int(karma) + 1) if karma > 0 or Beta(int(-karma) + 1, 1) otherwise. 

        Args:
            sides (int, optional): Number of sides of the rolled die. Defaults to 20.

        Returns:
            roll_outcome (int): The result of the roll.
        """
        

        # rolls = np.random.randint(1, sides + 1, size = 1 + int(abs(self.karma)))
        # if(self.karma >= 0):
        #     roll_outcome = min(rolls)
        # else:
        #     roll_outcome = max(rolls)
        if (self.karma >= 0):
            roll = np.random.beta(1, int(self.karma) + 1)
        else:
            roll = np.random.beta(1 + int(-self.karma), 1)

        roll_outcome = int(roll * sides) + 1

        # (roll_outcome * 2 - 1 - sides) / float(sides - 1)
        self.registerRoll((roll_outcome - 1) / float(sides - 1))
        return roll_outcome
    
    def slidingAdvatageBasedRolls(self, sides = 20):
        """Perform a roll that is like `advantageBasedRolls`, but takes into account fractional parts of karma.

        Args:
            sides (int, optional): Number of sides of the rolled die. Defaults to 20.

        Returns:
            roll_outcome (int): The result of the roll.
        """
        if (self.karma >= 0):
            roll = np.random.beta(1, self.karma + 1)
        else:
            roll = np.random.beta(1 - self.karma, 1)

        self.registerRoll(roll)

        return int((roll * sides) + 1)

    def pertModelBasedRoll(self, sides = 20, gamma = 4.0):
        """Perform a roll that is based on the general PERT model.
        logistic(-karma) will be set as the target "mode."

        Args:
            sides (int, optional): Number of sides of the rolled die. Defaults to 20.

        Returns:
            roll_outcome (int): The result of the roll.
        """

        mode = 1 / (1 + np.exp(self.karma))
        alpha = mode * gamma + 1
        beta = gamma + 2 - alpha
        roll = np.random.beta(alpha, beta)
        
        self.registerRoll(roll)
        return int((roll * sides) + 1)

    def wignerRoll(self, sides = 20):
        """These are rolls using the pertModelBasedRolls, but setting gamma to 1.0
        These are related to the Wigner Semicircle distribution. In particular, with karma = 0, this will sample exactly this distribution.
        For other karma cases, they will sample Beta(x, 3 - x), where x ∈ (1, 2)

        Args:
            sides (int, optional): Number of sides of the rolled die. Defaults to 20.
            
        Returns:
            roll_outcome (int): The result of the roll.
        """
        return self.pertModelBasedRoll(sides, gamma = 1.0)


    def interestingRoll(self, sides = 20, k = (np.sqrt(5) - 1)/2.0):
        """
        Roll based on x ∈ (k, 1 + k), and Beta(x, 1 + k*2 - x). Inspired by wignerRoll, but offsetting the minimum the beta parameters. k = 1 will be the same as wignerRoll.

        Empirically, with the default arguement of k = phi - 1, this appears to generate approximately every roll with equal distribution over time while tracking
        karma. (phi being the golden ratio)

        Args:
            sides (int, optional): Number of sides of the rolled die. Defaults to 20.
            k (float, optional): See discription above. 
        Returns:
            roll_outcome (int): The result of the roll.
        """
        assert(k >= 0)
        
        weighing = 1 / (1 + np.exp(self.karma))
        alpha = weighing + k
        beta = 1 + k + k - alpha
        roll = np.random.beta(alpha, beta)
        
        self.registerRoll(roll)
        return int((roll * sides) + 1)

if __name__ == "__main__":
    sides = 20
    kr = SimpleArmaRoller(-.5, np.array([.5, .25]))
    # kr = SimpleArmaRoller(0.0, 0.0)
    res = [kr.roll(sides) for i in range(20000)]
    plt.hist(np.diff(res), bins=sides)
    # plt.hist(list(np.diff(np.where(np.array(res) == 20))))
    plt.show()
