from scipy.optimize import minimize


class LinearModelsOptimizer(object):
    def __init__(
        self,
        row,
        x_ids,
        x0,
        models,
        bounds=None,
        constraints="pos",
        f_min=None,
    ):
        self.row = row
        self.x_ids = x_ids
        self.x0 = x0
        self.models = models
        self.bounds = bounds
        self.f_min = f_min
        if f_min is None:
            self.f_min = [0.0] * len(models)
        assert len(self.f_min) == len(models)
        if constraints == "pos":
            self.constraints = [
                {
                    "type": "ineq",
                    "fun": lambda x, model: self.f(x, model) - self.f_min[i],
                    "args": (model,),
                }
                for i, model in enumerate(self.models)
            ]
        else:
            self.constraints = constraints

    def f(self, x, model):
        X = self.row.copy()
        X[:, self.x_ids] = x
        return X @ model.coef_.T + model.intercept_

    def objective(self, x):
        return sum(self.f(x, model) for model in self.models)

    def jacobian(self, x):
        return sum(model.coef_[self.x_ids] for model in self.models)

    def optimize(self):
        return minimize(
            self.objective,
            self.x0,
            jac=self.jacobian,
            constraints=self.constraints,
            bounds=self.bounds,
        )
