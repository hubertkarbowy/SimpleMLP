package uladzislauharbuz;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;
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
                Ellipse2D.Double circle = new Ellipse2D.Double(p.x, p.y, diameter, diameter);
                g2d.fill(circle);
            }
        }
    }
}
