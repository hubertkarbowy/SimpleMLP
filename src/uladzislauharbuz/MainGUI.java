package uladzislauharbuz;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.text.NumberFormatter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import pl.hubertkarbowy.MLPNetwork;
import pl.hubertkarbowy.SerializedModel;

import static pl.hubertkarbowy.Utils.imgToInputs;

public class MainGUI {
    private JTabbedPane MainTabs;
    private JPanel mainPanel;
    private JPanel TrainTab;
    private JPanel RunTab;
    private JTextField trainFilePathField;
    private JButton openTrainSetButton;
    private JTextArea authorText;
    private JButton trainButton;
    private JSpinner imageWidthSpinner;
    private JSpinner imageHeightSpinner;
    private JList layersList;
    private JSpinner layerNeuronsSpinner;
    private JButton addLayerButton;
    private JButton deleteLayerButton;
    private JButton createNetworkButton;
    private JButton saveModelButton;
    private JButton openModelButton;
    private JLabel netStatusLabel;
    private JLabel trainStatusLabel;
    private JTextArea stdoutTextArea;
    private JPanel drawingPanel;
    private JTextField numberDetectedField;
    private JButton clearButton;
    private JButton numberDetectedLabel;
    private JDrawingArea drawingArea;

    private JFileChooser trainFileChooser = new JFileChooser();
    private File trainDirectory;
    private File saveTrainFile;
    private DefaultListModel layerListModel = new DefaultListModel();

    private final JSpinner[] SPINNERS = {imageHeightSpinner, imageWidthSpinner, layerNeuronsSpinner};
    private final JComponent[] CONTROLS_TO_DISABLE = {imageWidthSpinner, imageHeightSpinner, layerNeuronsSpinner, addLayerButton, createNetworkButton, layersList, deleteLayerButton};

    private MLPNetwork net;
    {   /* Disallow character input for spinners*/
        for (JSpinner spinner : SPINNERS) {
            JFormattedTextField imgF = ((JSpinner.NumberEditor) spinner.getEditor()).getTextField();
            ((NumberFormatter)imgF.getFormatter()).setAllowsInvalid(false);
            ((NumberFormatter)imgF.getFormatter()).setMinimum(1);
            spinner.setValue(1);
        }
    }

    private void updateFirstLayer() {
        layerListModel.setElementAt((int)imageHeightSpinner.getValue() * (int)imageWidthSpinner.getValue(), 0);
    }

    private void initComponents() {
        // Set model for LayersList and initialize it with a starting value
        layerListModel.addElement((int)imageHeightSpinner.getValue() * (int)imageWidthSpinner.getValue());
        layersList.setModel(layerListModel);

        // Set up the drawing canvas
        drawingArea = new JDrawingArea();
        drawingArea.setSize(250,250);
        drawingArea.setBackground(Color.BLACK);
        drawingArea.setVisible(true);
        drawingPanel.add(drawingArea, BorderLayout.CENTER);
        drawingPanel.revalidate();
    }

    public MainGUI() {

        initComponents();

        PrintStream stdoutStream = new PrintStream(new JTextAreaOutputStream(stdoutTextArea));
        System.setOut(stdoutStream);
        System.setErr(stdoutStream);

        openTrainSetButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                trainFileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
                int ret = trainFileChooser.showOpenDialog(null);
                if (ret == trainFileChooser.APPROVE_OPTION) {
                    trainDirectory = trainFileChooser.getSelectedFile();
                    trainFilePathField.setText(trainDirectory.getPath());
                }
            }
        });
        addLayerButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                layerListModel.addElement(layerNeuronsSpinner.getValue());
            }
        });
        imageWidthSpinner.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                updateFirstLayer();
            }
        });
        imageHeightSpinner.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                updateFirstLayer();
            }
        });
        deleteLayerButton.addActionListener(e -> {
            int selectedItem = layersList.getSelectedIndex();
            if (selectedItem > 0) {
                layerListModel.remove(selectedItem);
            }
        });
        createNetworkButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                int[] layers = new int[layerListModel.size()];
                for (int i = 0; i < layerListModel.size(); ++i) {
                    layers[i] = (int)layerListModel.getElementAt(i);
                }
                if (layers.length < 2) {
                    JOptionPane.showMessageDialog(null, "Please define at least the input and output layers.");
                    return;
                }
                if (layers[layers.length-1] != 1) {
                    JOptionPane.showMessageDialog(null, "Not a binary classification network. Last layer must have a single output.");
                    return;
                }
                net = new MLPNetwork(layers);
                Arrays.stream(CONTROLS_TO_DISABLE).forEach(c -> c.setEnabled(false));
                netStatusLabel.setText("<html><font color=green><b>READY TO TRAIN</b></font></html>");
            }
        });

        trainButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (net == null) {
                    JOptionPane.showMessageDialog(null, "Please build a model first.");
                    return;
                }
                if (trainDirectory == null || !trainDirectory.exists() || !trainDirectory.isDirectory()) {
                    JOptionPane.showMessageDialog(null, "Please choose a directory containing 'pos' and 'neg' subdirectories.");
                    return;
                }
                try {
                    net.setInputs(trainDirectory);
                } catch (IOException ex) {
                    trainStatusLabel.setText("Failed to train network: " + ex.getMessage());
                }
                Thread thread = new Thread(() -> { net.train(); saveModelButton.setEnabled(true);});
                thread.start();
            }
        });
        saveModelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                trainFileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
                int ret = trainFileChooser.showSaveDialog(null);
                if (ret == trainFileChooser.APPROVE_OPTION) {
                    saveTrainFile = trainFileChooser.getSelectedFile();
                }
                try {
                    net.saveModel(saveTrainFile.getPath());
                } catch (IOException ex) {
                    System.out.println(ex.getMessage());
                }
            }
        });
        openModelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                trainFileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
                int ret = trainFileChooser.showOpenDialog(null);
                if (ret == trainFileChooser.APPROVE_OPTION) {
                    try {
                        net = MLPNetwork.restoreModel(trainFileChooser.getSelectedFile().getPath());
                        int[] neurons = net.getNeurons();
                        layerListModel.clear();
                        Arrays.stream(neurons).forEach(l -> layerListModel.addElement(l));
                        int inferredWidthAndHeight = (int) Math.sqrt(neurons[0]);
                        imageHeightSpinner.setValue(inferredWidthAndHeight);
                        imageWidthSpinner.setValue(inferredWidthAndHeight);
                        Arrays.stream(CONTROLS_TO_DISABLE).forEach(c -> c.setEnabled(false));
                        netStatusLabel.setText("<html><font color=blue><b>RESTORED FROM FILE</b></font></html>");
                    } catch (Exception ex) {
                        ex.printStackTrace();
                        System.out.println(ex.getMessage());
                    }
                }
            }
        });
        drawingArea.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                super.mouseDragged(e);
                drawingArea.setDraw(true);
                drawingArea.setCursor(e.getPoint(), 5);
                drawingArea.repaint();
                BufferedImage img = drawingArea.getImage
                        ( (int)imageWidthSpinner.getValue()
                        , (int)imageHeightSpinner.getValue());
                float[] inputs = imgToInputs(img);
                if (net.predictBinary(inputs)) {
                    numberDetectedField.setText("3");
                } else {
                    numberDetectedField.setText("Not 3");
                }
            }
        });

        clearButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                drawingArea.setDraw(false);
                drawingArea.clearPaint();
                drawingArea.repaint();
            }
        });
    }

    public static void main(String args[]) {
        JFrame mainFrame = new JFrame("Simple MLP");
        mainFrame.setContentPane(new MainGUI().mainPanel);
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        mainFrame.pack();
        mainFrame.setLocationRelativeTo(null);
        mainFrame.setVisible(true);
    }
}
