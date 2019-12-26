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
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import pl.hubertkarbowy.MLPNetwork;

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
    private JTextField textField1;
    private JButton clearButton;
    private JDrawingArea drawingArea;

    private JFileChooser trainFileChooser = new JFileChooser();
    private File trainFile;
    private File saveTrainFile;
    private DefaultListModel layerListModel = new DefaultListModel();
    private List<JSpinner> spinners = new ArrayList<>();

    private MLPNetwork net;

    private void updateFirstLayer() {
        layerListModel.setElementAt((int)imageHeightSpinner.getValue() * (int)imageWidthSpinner.getValue()
                , 0);
    }

    public MainGUI() {
        spinners.add(imageHeightSpinner);
        spinners.add(imageWidthSpinner);
        spinners.add(layerNeuronsSpinner);

        /* Disallow character input for spinners*/
        for (JSpinner spinner : spinners) {
            JFormattedTextField imgF = ((JSpinner.NumberEditor) spinner.getEditor()).getTextField();
            ((NumberFormatter)imgF.getFormatter()).setAllowsInvalid(false);
            ((NumberFormatter)imgF.getFormatter()).setMinimum(1);
            spinner.setValue(1);
        }

        layerListModel.addElement((int)imageHeightSpinner.getValue() * (int)imageWidthSpinner.getValue());
        layersList.setModel(layerListModel);

        PrintStream stdoutStream = new PrintStream(new JTextAreaOutputStream(stdoutTextArea));
        System.setOut(stdoutStream);
        System.setErr(stdoutStream);

        drawingArea = new JDrawingArea();
        drawingArea.setSize(250,250);
        drawingArea.setBackground(Color.WHITE);
        drawingArea.setVisible(true);

        openTrainSetButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                trainFileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
                int ret = trainFileChooser.showOpenDialog(null);
                if (ret == trainFileChooser.APPROVE_OPTION) {
                    trainFile = trainFileChooser.getSelectedFile();
                    trainFilePathField.setText(trainFile.getPath());
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
        deleteLayerButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                int selectedItem = layersList.getSelectedIndex();
                if (selectedItem > 0) {
                    layerListModel.remove(selectedItem);
                }
            }
        });
        createNetworkButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                int[] layers = new int[layerListModel.size() + 1];
                for (int i = 0; i < layerListModel.size(); ++i) {
                    layers[i] = (int)layerListModel.getElementAt(i);
                }
                layers[layers.length-1] = 1;

                net = new MLPNetwork(layers);
                netStatusLabel.setText("OK");
            }
        });

        trainButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    net.setInputs(trainFile);
                } catch (IOException ex) {
                    trainStatusLabel.setText("Failed to train network: " + ex.getMessage());
                }
                Thread thread = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        net.train();
                    }
                });
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
                        net = new MLPNetwork(new int[0]);
                        net.restoreModel(trainFileChooser.getSelectedFile().getPath());
                    } catch (Exception ex) {
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
            }
        });
        drawingPanel.add(drawingArea);
        drawingPanel.revalidate();
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
        mainFrame.setVisible(true);
    }
}
