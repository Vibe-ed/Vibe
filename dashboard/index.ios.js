/* NOTES
Have API structure setup, but they are not used.
Investigate putting image into a 'top contaner'
change individual bar colore in bar chart
*/

/* Libraries */
import React, {
  AppRegistry,
  Component,
  Image,
  ListView,
  StyleSheet,
  Text,
  View,
  Animated
} from 'react-native';
import RNChart from 'react-native-chart';

/* API Information */
/* NOT USED - PLACE HOLDER */
var API_KEY = '7waqfqbprs7pajbz28mqf6vz';
var API_URL = 'http://api.rottentomatoes.com/api/public/v1.0/lists/movies/in_theaters.json';
var PAGE_SIZE = 25;
var PARAMS = '?apikey=' + API_KEY + '&page_limit=' + PAGE_SIZE;
var REQUEST_URL = API_URL + PARAMS;

/* Styles */
var styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'white',
    },
    chart: {
        position: 'absolute',
        //top: 16,
        height: 200,
        left: 4,
        bottom: 4,
        right: 16,
    },
    image: {
        height: 436,
        left: 4,
        right: 4,
        top: 20,
        position: 'absolute',
    }
});

var chartAPI = {
    users: ['henry', 'yael', 'tyler'],
    proportions: [0.01, 0.98, 0.01],
    commLevel: 2
};

var chartData = [
    {
        name: 'BarChart',
        type: 'bar',
        color:'purple',
        widthPercent: 0.6,
        data: chartAPI.proportions,
    }
];

var xLabels = chartAPI.users

var commURL = ""
switch (chartAPI.commLevel) {
  case 3:
    commURL = 'https://media.giphy.com/media/cz70wJgrvLa9i/giphy.gif';
    break;
  case 2:
    commURL = 'http://rs138.pbsrc.com/albums/q249/xxxANBUxxx/o.gif~c200'
    break;
  case 1:
    commURL = 'https://media.giphy.com/media/OOdKuvLmj8QYU/giphy.gif'
    break;
  default:
    commURL = 'https://media.giphy.com/media/cz70wJgrvLa9i/giphy.gif';

}


class dashboard extends React.Component {
  /* Inital state when loading data */
  constructor(props: any) {
    super(props);
    this.state = {
      graph: null
    };
  }

  /* Fetches the data exactly once after component finishes loading */
//  componentDidMount() {
    /* Fetch chart data */
//    this.fetchData();
    /* Animate image */
//    this.state.bounceValue.setValue(1.5);     // Start large
//    Animated.spring(                          // Base: spring, decay, timing
//      this.state.bounceValue,                 // Animate `bounceValue`
//      {
//        toValue: 0.8,                         // Animate to smaller size
//        friction: 1,                          // Bouncier spring
//      }
//    ).start();                                // Start the animation
//  }


  /* NOT NEEDED YET */
  /* Function to actually fetch data */
  /*
  fetchData() {
    fetch(REQUEST_URL)
      .then((response) => response.json())
      .then((responseData) => {
        this.setState({
          graph: responseData.graph,
        });
      })
      .done();
  } */

  /* Master render function. Will render image or loading screen if
     trouble pinging server. */
  render() {
    /* Dislpays loading text if data not loaded */
    /* NOT NEEDED
    if (!this.state.loaded) {
      return this.renderLoadingView();
    } */
    return (
      <View style={styles.container}>
        <RNChart style={styles.chart}       // Render the graph
                 chartData={chartData}
                 verticalGridStep={5}
                 xLabels={xLabels}
             />
        <Animated.Image                    // Render the image
                 source={{uri: commURL}}
                 style={styles.image}
             />
      </View>
    );
  }

  /* Renders loading text */
  /* CURRENTLY NOT USED */
  renderLoadingView() {
    return (
      <View style={styles.container}>
        <Text>
          Loading data...
        </Text>
      </View>
    );
  }

  /* Renders the graph */
  renderGraph() {
    return (
      <View style={styles.container}>
        <RNChart style={styles.chart}
                 chartData={chartData}
                 verticalGridStep={5}
                 xLabels={xLabels}
             />
        </View>
    );
  }

  /* Render the image */
/*  renderImage(): ReactElement {
    return (
      <Animated.Image                         // Base: Image, Text, View
        source={{uri: 'http://i.imgur.com/XMKOH81.jpg'}}
        style={styles.image}
      />
    );
  } */
}

AppRegistry.registerComponent('dashboard', () => dashboard);
