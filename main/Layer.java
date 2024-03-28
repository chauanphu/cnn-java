public class Layer {
    Tensor2D weights;
    Tensor2D intercept;
    int nNodes;
    Tensor2D output;
    Tensor2D input;

    Layer(int nNodes) {
        this.nNodes = nNodes;
    }

    public void initiate_weights (int row, int col) {
        this.weights = Tensor2D.randoms(row, col);
    }

    public void initiate_intercept (int row, int col) {
        this.intercept = Tensor2D.randoms(row, col);
    }

    
    public Tensor2D forward() {
        // z = W * x + b; z's dimensions = [n_curr, m], W's dimensions =  [n_curr, n_prev], b's dimensions = [n_curr, m]
        Tensor2D z = this.weights.dot(input).add(intercept);
        this.output = z;
        return z;
    }

    @Override
    public String toString() {
        return "Number of nodes: " + nNodes;
    }
}