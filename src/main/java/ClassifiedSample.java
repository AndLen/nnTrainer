
import java.util.List;

/**
 * Created by Andrew on 28/03/2015.
 */
public class ClassifiedSample extends Sample {

    private final SampleLocation sampleLocation;

    public ClassifiedSample(SampleLocation sampleLocation, String bssid, Double frequency, List<Double> rssiReadings) {
        super(bssid, frequency,rssiReadings);
        this.sampleLocation = sampleLocation;
    }

    public SampleLocation getSampleLocation() {
        return sampleLocation;
    }


    public static class SampleLocation {
        private final double x;
        private final double y;

        public SampleLocation(double x, double y) {
            this.x = x;
            this.y = y;
        }

        public double getX() {
            return x;
        }

        public double getY() {
            return y;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            SampleLocation that = (SampleLocation) o;

            if (Double.compare(that.x, x) != 0) return false;
            if (Double.compare(that.y, y) != 0) return false;

            return true;
        }

        @Override
        public int hashCode() {
            int result;
            long temp;
            temp = Double.doubleToLongBits(x);
            result = (int) (temp ^ (temp >>> 32));
            temp = Double.doubleToLongBits(y);
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            return result;
        }
    }
}
