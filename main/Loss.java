public abstract class Loss {
    /**
     * Calculate the loss
     * @param y The one-hot encoding matrix of the target values (true values), dimensions = [n, m], where n is the number of classes and m is the number of samples
     * @param y_hat The probability distribution of predicted values, dimensions = [n, m], where n is the number of classes and m is the number of samples
     * @return
     */
    public abstract Tensor2D cal(Tensor2D y, Tensor2D y_hat);
}

class CategoryCrossEntropy extends Loss {
    CategoryCrossEntropy() {
    }
    
    @Override
    public Tensor2D cal(Tensor2D y, Tensor2D y_hat) {
        Tensor2D res = new Tensor2D(1, y.ncols);
        for (int i  = 0; i < y_hat.ncols; i++) {
            double loss = 0;
            for (int j = 0; j < y_hat.nrows; j++) {
                loss -= ;
            }   
        }
        return res;
    }
}