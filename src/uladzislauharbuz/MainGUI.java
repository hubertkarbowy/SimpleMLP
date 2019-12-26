package uladzislauharbuz;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.text.NumberFormatter;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
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
    private JTextField trainSaveTextField;
    private JButton trainSaveFilePathButton;
    private JButton trainSaveButton;
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
    private JLabel saveTrainStatusLabel;
    private JLabel trainStatusLabel;
    private JTextArea stdoutTextArea;

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
        trainSaveFilePathButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                trainFileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
                int ret = trainFileChooser.showSaveDialog(null);
                if (ret == trainFileChooser.APPROVE_OPTION) {
                    saveTrainFile = trainFileChooser.getSelectedFile();
                    trainSaveTextField.setText(saveTrainFile.getPath());
                }
            }
        });
        trainSaveButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    net.saveModel(saveTrainFile.getPath());
                } catch (IOException ex) {
                    saveTrainStatusLabel.setText("Failed to save trained network:" + ex.getMessage());
                }
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
    }

    public static void main(String args[]) {
        JFrame mainFrame = new JFrame("Simple MLP");
        mainFrame.setContentPane(new MainGUI().mainPanel);
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        mainFrame.pack();
        mainFrame.setVisible(true);
    }

}
