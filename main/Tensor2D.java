public class Tensor2D {
    public int nrows;
    public int ncols;
    public double[][] data;

    Tensor2D (int nrows, int ncols) {
        this.nrows = nrows;
        this.ncols = ncols;
        this.data = new double[nrows][ncols];
    }
    Tensor2D(double[][] data) {
        this.nrows = data.length;
        this.ncols = data[0].length;
        this.data = data;
    }
    public Tensor2D fill_data(double[][] data) {
        // Check dimension of data
        if ((data.length != this.nrows) || (data[0].length != this.ncols)) {
            throw new IllegalArgumentException("The dimensions of the data must be the same as the tensor.");
        }
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) this.data[i][j] = data[i][j];
        }
        return this;
    }

    public Tensor2D update(double scala, int row, int col) {
        this.data[row][col] = scala;
        return this;
    }

    public Tensor2D transpose() {
        Tensor2D res = new Tensor2D(this.ncols, this.nrows);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) res.data[j][i] = this.data[i][j];
        }
        return res;
    }

    public static Tensor2D randoms(int nrows, int ncols) {
        Tensor2D res = new Tensor2D(nrows, ncols);
        for (int i = 0; i < nrows; i++) {
            // Fill the tensor with random numbers from -1 to 1
            for (int j = 0; j < ncols; j++) res.data[i][j] = Math.random() * 2 - 1;
        }
        return res;
    }
    
    public Tensor2D add (Tensor2D otherTensor) {
        if ((otherTensor.nrows != this.nrows) || (otherTensor.ncols != this.ncols)) {
            throw new IllegalArgumentException("The dimensions of the two tensors must be the same.");
        }
        Tensor2D res = new Tensor2D(this.nrows, this.ncols);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) res.data[i][j] += otherTensor.data[i][j];
        }
        return res;
    }

    public Tensor2D mul (Tensor2D otherTensor) {
        if ((otherTensor.nrows != this.nrows) || (otherTensor.ncols != this.ncols)) {
            throw new IllegalArgumentException("The dimensions of the two tensors must be the same.");
        }
        Tensor2D res = new Tensor2D(this.nrows, this.ncols);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) res.data[i][j] *= otherTensor.data[i][j];
        }
        return res;
    }
    
    public Tensor2D dot (Tensor2D otherTensor) {
        if (this.ncols != otherTensor.nrows)  {
            throw new IllegalArgumentException("The number of columns of the first tensor must be the same as the number of rows of the second tensor.");
        }
        Tensor2D res = new Tensor2D(this.nrows, otherTensor.ncols);
        for (int i = 0; i < nrows; i++){
            for (int j = 0; j < otherTensor.ncols; j++) {
                for (int k = 0; k < ncols; k++) res.data[i][j] += this.data[i][k] * otherTensor.data[k][j];
            }
        }

        return res;
    }
   
    @Override
    public String toString() {
        String res = "";
        for (int i = 0; i < this.nrows; i++) {
            for (int j = 0; j < this.ncols; j++) {
                res += this.data[i][j] + "\t";
            }
            res += "\n";
        }
        return res;
    }

    public static void main(String[] args) {
        Tensor2D t = new Tensor2D(2, 3);
        double[][] data = {{1, 2, 3}, {4, 5, 6}};
        t.fill_data(data);
        Tensor2D t2 = new Tensor2D(3,2);
        double[][] data2 = {{7, 8}, {9, 10}, {11, 12}};
        t2.fill_data(data2);
        System.out.println(t.dot(t2));
    }
}