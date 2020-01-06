package uladzislauharbuz;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.image.BufferedImage;
import java.util.LinkedList;

public class JDrawingArea extends JPanel {

    private Point cursor;
    private int diameter;
    private boolean draw = false;
    private LinkedList<Point> paint = new LinkedList<>();

    public void setCursor(Point x, int diameter) {
        cursor = x;
        this.diameter = diameter;
    }

    public void setDraw(boolean x) {
        draw = x;
    }

    public void clearPaint() {
        paint.clear();
    }

    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (draw) {
            paint.add(cursor);
            for (Point p : paint) {
                Graphics2D g2d = (Graphics2D) g;
                g2d.setColor(Color.WHITE);
                Ellipse2D.Double circle = new Ellipse2D.Double(p.x, p.y, diameter, diameter);
                g2d.fill(circle);
            }
        }
    }

    public BufferedImage getImage() {
        BufferedImage bi = new BufferedImage(this.getWidth(), this.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = bi.createGraphics();
        this.paint(g);
        return bi;
    }

    public BufferedImage getImage(int width, int height) {
        BufferedImage img = getImage();
        BufferedImage rescaled = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = rescaled.createGraphics();
        g2d.drawImage(img, 0, 0, width, height, null);
        g2d.dispose();
        return rescaled;
    }
}
