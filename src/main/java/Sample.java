import java.util.List;

/**
 * Created by Andrew on 28/03/2015.
 */
public class Sample {
    private final String bssid;
    private Double frequency;
    private final List<Double> rssiReadings;

    public Sample(String bssid, Double frequency, List<Double> rssiReadings){
        this.bssid = bssid;
        this.frequency = frequency;
        this.rssiReadings = rssiReadings;
    }

    public String getBssid() {
        return bssid;
    }

    public List<Double> getRssiReadings() {
        return rssiReadings;
    }

    public Double getAverageRSSI(){
        double sum = 0;
        for (Double rssiReading : rssiReadings) {
            sum+=rssiReading;
        }
        return sum/rssiReadings.size();

    }
}
