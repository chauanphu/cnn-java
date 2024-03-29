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
        List<Tensor2D> data = readCSV("main/mnist-train.csv", true);
        Tensor2D inputTensor2d = data.get(0);
        Tensor2D targetTensor2d = data.get(1);
        System.out.println(inputTensor2d.ncols + " " + targetTensor2d.ncols);

        // Create the network
        Tensor2D input = new Tensor2D(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });
        Sequential model = new Sequential(
            new Layer[] {
                new InputLayer(784),
                new DenseLayer(128, Activation.ActivationType.RELU),
                new DenseLayer(10, Activation.ActivationType.SOFTMAX)
            }
        );
        model.summary();
        model.fit(inputTensor2d, targetTensor2d, 10, 0.01);
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

    public static List<Tensor2D> readCSV(String filename, boolean hasHead) {
        List<int[]> data = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        List<Tensor2D> result = new ArrayList<>();
        // Try to read the CSV file
        try (Scanner scanner = new Scanner(new File(filename))) {
            if (hasHead) {
                scanner.nextLine();
            }
            while (scanner.hasNextLine()) {
                String[] line = scanner.nextLine().split(",");
                // Skip the index column
                labels.add(Integer.parseInt(line[0]));
                int[] row = new int[line.length - 1];
                for (int i = 1; i < line.length - 1; i++) {
                    row[i] = Integer.parseInt(line[i]);
                }
                data.add(row);
            }
            result.add(new Tensor2D(data));
            Tensor2D target = new Tensor2D(labels.size(), 10);
            for (int i = 0; i < labels.size(); i++) {
                target.data[i][labels.get(i)] = 1;
            }
            result.add(target);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return result;
    }
}
