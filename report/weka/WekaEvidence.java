import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.imageio.ImageIO;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JTabbedPane;
import javax.swing.SwingUtilities;

import weka.core.Instances;
import weka.core.Version;
import weka.core.converters.ConverterUtils.DataSource;
import weka.gui.AttributeSelectionPanel;
import weka.gui.explorer.Explorer;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

/** Xuất ảnh từ panel thật của Weka; mọi dữ liệu và điểm ROC do Python chuẩn bị. */
public final class WekaEvidence {
    private static final int WIDTH = 1200;
    private static final int HEIGHT = 740;

    private static AttributeSelectionPanel findAttributeList(Container container) {
        for (Component child : container.getComponents()) {
            if (child instanceof AttributeSelectionPanel) {
                return (AttributeSelectionPanel) child;
            }
            if (child instanceof Container) {
                AttributeSelectionPanel nested = findAttributeList((Container) child);
                if (nested != null) {
                    return nested;
                }
            }
        }
        return null;
    }

    private static Explorer preprocess(Path path, String selectedAttribute) throws Exception {
        Instances instances = DataSource.read(path.toString());
        instances.setClassIndex(instances.attribute("Potability").index());
        Explorer explorer = new Explorer();
        explorer.getPreprocessPanel().setInstances(instances);
        AttributeSelectionPanel attributes = findAttributeList(explorer.getPreprocessPanel());
        if (attributes == null) {
            throw new IllegalStateException("Weka Preprocess attribute list not found");
        }
        int column = instances.attribute(selectedAttribute).index();
        attributes.getSelectionModel().setSelectionInterval(column, column);
        return explorer;
    }

    private static VisualizePanel plot(Path path, boolean roc) throws Exception {
        Instances instances = DataSource.read(path.toString());
        PlotData2D points = new PlotData2D(instances);
        points.setPlotName(roc ? "Python Naive Bayes ROC (class 1)" : "Python Naive Bayes test outcomes");
        if (roc) {
            boolean[] connect = new boolean[instances.numInstances()];
            int[] size = new int[instances.numInstances()];
            for (int index = 0; index < connect.length; index++) {
                connect[index] = index > 0;
                size[index] = 1;
            }
            points.setConnectPoints(connect);
            points.setShapeSize(size);
            points.setCustomColour(new Color(32, 81, 154));
        }
        VisualizePanel panel = new VisualizePanel();
        panel.setShowAttBars(false);
        panel.setShowClassPanel(!roc);
        panel.setMasterPlot(points);
        panel.setXIndex(0);
        panel.setYIndex(1);
        if (!roc) {
            panel.setColourIndex(2);
        }
        return panel;
    }

    private static void layout(Container container) {
        container.doLayout();
        for (Component child : container.getComponents()) {
            if (child instanceof Container) {
                layout((Container) child);
            }
        }
    }

    private static BufferedImage paintPanel(JComponent panel) {
        panel.setSize(WIDTH, HEIGHT);
        layout(panel);
        BufferedImage image = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        try {
            graphics.setColor(Color.WHITE);
            graphics.fillRect(0, 0, WIDTH, HEIGHT);
            panel.paint(graphics);
        } finally {
            graphics.dispose();
        }
        return image;
    }

    private static void exportPanel(JComponent panel, Path destination) throws IOException {
        if (!ImageIO.write(paintPanel(panel), "PNG", destination.toFile())) {
            throw new IOException("PNG exporter unavailable");
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1 || args.length > 2 || (args.length == 2 && !args[1].equals("--show"))) {
            throw new IllegalArgumentException("Usage: WekaEvidence ARFF_DIRECTORY [--show]");
        }
        Path directory = Paths.get(args[0]);
        Map<String, JComponent> panels = new LinkedHashMap<>();
        SwingUtilities.invokeAndWait(() -> {
            try {
                panels.put("01-data-and-class", preprocess(directory.resolve("raw.arff"), "Potability"));
                panels.put("02-development-missing", preprocess(directory.resolve("development_raw.arff"), "Sulfate"));
                panels.put("03-development-processed", preprocess(directory.resolve("development_processed.arff"), "Sulfate"));
                panels.put("04-test-partition", preprocess(directory.resolve("test_processed.arff"), "Potability"));
                panels.put("05-roc", plot(directory.resolve("roc.arff"), true));
                panels.put("06-errors", plot(directory.resolve("errors.arff"), false));
                panels.put("07-development-class", preprocess(directory.resolve("development_raw.arff"), "Potability"));
                panels.put("08-ph-processed", preprocess(directory.resolve("development_processed.arff"), "ph"));
                for (JComponent panel : panels.values()) {
                    // Lượt vẽ đầu kích hoạt phép dựng histogram nền của Weka.
                    paintPanel(panel);
                }
            } catch (Exception error) {
                throw new IllegalStateException(error);
            }
        });
        // Cho các tác vụ histogram của Weka hoàn thành trước khi xuất đồ họa.
        Thread.sleep(850);
        SwingUtilities.invokeAndWait(() -> {
            try {
                for (Map.Entry<String, JComponent> panel : panels.entrySet()) {
                    exportPanel(panel.getValue(), directory.resolve(panel.getKey() + ".png"));
                }
            } catch (IOException error) {
                throw new IllegalStateException(error);
            }
            if (args.length == 2) {
                JTabbedPane tabs = new JTabbedPane();
                panels.forEach(tabs::addTab);
                JFrame window = new JFrame("Weka " + Version.VERSION + " - Water Potability");
                window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                window.setContentPane(tabs);
                window.setSize(WIDTH, HEIGHT + 60);
                window.setLocationByPlatform(true);
                window.setVisible(true);
            }
        });
        System.out.println("Weka " + Version.VERSION + ": " + panels.size() + " native panel images exported.");
        if (args.length == 1) {
            System.exit(0);
        }
    }
}
