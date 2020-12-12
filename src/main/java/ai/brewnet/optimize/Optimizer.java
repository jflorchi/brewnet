package ai.brewnet.optimize;

import ai.brewnet.Matrix2D;

public abstract class Optimizer {

    public abstract Matrix2D function(Matrix2D x, Matrix2D weights);

    public abstract Matrix2D functionDerivative(Matrix2D x, Matrix2D weights);

}
