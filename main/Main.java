import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import javax.imageio.ImageIO;

public class Main {
    public static void main(String[] args) {
        // Load the dataset
        // int[][] data = readCSV("main/mnist-train.csv", true);
        // Tensor2D inputTensor2d = new Tensor2D(data);
        // System.out.println(inputTensor2d.nrows + " " + inputTensor2d.ncols);

        // Create the network
        Tensor2D input = new Tensor2D(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });
        Sequential model = new Sequential(
            new Layer[] {
                new InputLayer(3, input),
                new DenseLayer(128),
                new DenseLayer(10)
            }
        );
        model.summary();
        model.predict(input);
        model.print_result();
    }

    public int[][] getMatrixOfImage(BufferedImage bufferedImage) {
        // Try to read the image file
        try {
            bufferedImage = ImageIO.read(new File("image.jpg"));
        } catch (Exception e) {
            e.printStackTrace();
        }
        int width = bufferedImage.getWidth(null);
        int height = bufferedImage.getHeight(null);
        int[][] pixels = new int[width][height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                pixels[i][j] = bufferedImage.getRGB(i, j);
            }
        }
    
        return pixels;
    }

    public static int[][] readCSV(String filename, boolean hasHead) {
    List<int[]> data = new ArrayList<>();
    // Try to read the CSV file
    try (Scanner scanner = new Scanner(new File(filename))) {
        if (hasHead) {
            scanner.nextLine();
        }
        while (scanner.hasNextLine()) {
            String[] line = scanner.nextLine().split(",");
            // Skip the index column
            int[] row = new int[line.length - 1];
            for (int i = 1; i < line.length - 1; i++) {
                row[i] = Integer.parseInt(line[i]);
            }
            data.add(row);
        }
    } catch (FileNotFoundException e) {
        e.printStackTrace();
    }

    // Convert list of rows to 2D array
    int[][] result = new int[data.size()][];
    for (int i = 0; i < data.size(); i++) {
        result[i] = data.get(i);
    }

    return result;
    }
}
