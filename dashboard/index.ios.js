/* NOTES
Have API structure setup, but they are not used.

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
    }
//    image: {
//        flex: 1,
//        transform: [                        // `transform` is an ordered array
//          {scale: this.state.bounceValue},  // Map `bounceValue` to `scale`
//      ]
//    }
});

var chartAPI = {
    users: ['henry', 'yael', 'tyler'],
    proportions: [0.01, 0.98, 0.01],
    commLevel: 3
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

class dashboard extends React.Component {
  /* Inital state when loading data */
  constructor(props: any) {
    super(props);
    this.state = {
//      bounceValue: new Animated.Value(0), // animation
      graph: null                         // graph
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
          graph: responseData.movies,
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
        <RNChart style={styles.chart}
                 chartData={chartData}
                 verticalGridStep={5}
                 xLabels={xLabels}
             />
        </View>
    );


//    return (
//      this.renderGraph
//      this.renderImage
//    );
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
