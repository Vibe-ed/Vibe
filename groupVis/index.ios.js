import React, {
  Component,
} from 'react';
import {
  AppRegistry,
  Image,
  ListView,
  StyleSheet,
  Text,
  View,
  Animated
} from 'react-native';

var graphAPI1 = {
  graph:  [[4, 1, 0, 0],
           [2, 2, 1, 0],
           [0, 0, 1, 1],
           [1, 1, 1, 1]]
}

var graphAPI2 = {
  users: ['henry', 'yael', 'tyler', 'tom'],
  numUsers: 4
}

var styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'white',
    },
    user0: {
      height: 60,
      left: 20,
      right: 275,
      top: 100,
      position: 'absolute',
    },
    user1: {
      height: 60,
      left: 275,
      right: 20,
      top: 100,
      position: 'absolute',
    },
    user2: {
      height: 60,
      left: 20,
      right: 275,
      bottom: 100,
      position: 'absolute',
    },
    user3: {
      height: 60,
      left: 275,
      right: 20,
      bottom: 100,
      position: 'absolute',
    },
});

var graphLabels = {
  users: graphAPI2.users
}

var commURL = 'https://facebook.github.io/react/img/logo_og.png'
class groupVis extends React.Component {
  /* Inital state when loading data */
  // constructor(props: any) {
  //   super(props);
  //   this.state = {
  //     graph: null
  //   };
  // }

  /* Fetches the data exactly once after component finishes loading */
  // componentDidMount() {
  //   /* Fetch chart data */
  //   this.fetchData();
  // }


  /* NOT NEEDED YET */
  /* Function to actually fetch data */
  // fetchData() {
  //   fetch(REQUEST_URL)
  //     .then((response) => response.json())
  //     .then((responseData) => {
  //       this.setState({
  //         graph: responseData.graph,
  //       });
  //     })
  //     .done();
  // }

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
        <Animated.Image                    // Render the image
                 source={{uri: commURL}}
                 style={styles.user0}
             />
        <Animated.Image                    // Render the image
                 source={{uri: commURL}}
                 style={styles.user1}
             />
        <Animated.Image                    // Render the image
                 source={{uri: commURL}}
                 style={styles.user2}
             />
        <Animated.Image                    // Render the image
                 source={{uri: commURL}}
                 style={styles.user3}
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
}

  /* Renders the graph */
  // renderGraph() {
  //   return (
  //     <View style={styles.container}>
  //       <RNChart style={styles.chart}
  //                chartData={chartData}
  //                verticalGridStep={5}
  //                xLabels={xLabels}
  //            />
  //       </View>
  //   );
  // }

// var graphData = {
//   user0: {
//     name: graphAPI2.users[0]
//     nodeSize: graphAPI1.graph[0][0]
//     }
// }

AppRegistry.registerComponent('groupVis', () => groupVis);

/* START OTHER PROJECT EXAMPLE */


// /* NOTES
// Have API structure setup, but they are not used.
// Investigate putting image into a 'top contaner'
// change individual bar color in bar chart
// timed refreshed/interact with restful API
// put on iphone
// write portion for ipad mic with portioning
// */
//
// /* Libraries */
// import React, {
//   AppRegistry,
//   Component,
//   Image,
//   ListView,
//   StyleSheet,
//   Text,
//   View,
//   Animated
// } from 'react-native';
// import RNChart from 'react-native-chart';
//
// /* Import timer */
// var TimerMixin = require('react-timer-mixin');
//
// /* API Information */
// /* NOT USED - PLACE HOLDER */
// var API_KEY = '7waqfqbprs7pajbz28mqf6vz';
// var API_URL = 'http://api.rottentomatoes.com/api/public/v1.0/lists/movies/in_theaters.json';
// var PAGE_SIZE = 25;
// var PARAMS = '?apikey=' + API_KEY + '&page_limit=' + PAGE_SIZE;
// var REQUEST_URL = API_URL + PARAMS;
//
// /* Styles */
// var styles = StyleSheet.create({
//     container: {
//         flex: 1,
//         justifyContent: 'center',
//         alignItems: 'center',
//         backgroundColor: 'white',
//     },
//     chart: {
//         position: 'absolute',
//         //top: 16,
//         height: 200,
//         left: 4,
//         bottom: 4,
//         right: 16,
//     },
//     image: {
//         height: 436,
//         left: 4,
//         right: 4,
//         top: 20,
//         position: 'absolute',
//     }
// });
//
// var chartAPI = {
//     users: ['henry', 'yael', 'tyler'],
//     proportions: [0.01, 0.4, 0.01],
//     commLevel: 1
// };
//
// var chartData = [
//     {
//         name: 'BarChart',
//         type: 'bar',
//         color:'purple',
//         widthPercent: 0.6,
//         data: chartAPI.proportions,
//     }
// ];
//
// var xLabels = chartAPI.users
//
// var commURL = ""
// switch (chartAPI.commLevel) {
//   case 3:
//     commURL = 'https://media.giphy.com/media/cz70wJgrvLa9i/giphy.gif';
//     break;
//   case 2:
//     commURL = 'http://rs138.pbsrc.com/albums/q249/xxxANBUxxx/o.gif~c200'
//     break;
//   case 1:
//     commURL = 'https://media.giphy.com/media/OOdKuvLmj8QYU/giphy.gif'
//     break;
//   default:
//     commURL = 'https://media.giphy.com/media/cz70wJgrvLa9i/giphy.gif';
// }
//
//
// class dashboard extends React.Component {
//   /* Inital state when loading data */
//   constructor(props: any) {
//     super(props);
//     this.state = {
//       graph: null
//     };
//   }
//
//   /* Fetches the data exactly once after component finishes loading */
//   componentDidMount() {
//     /* Fetch chart data */
//     this.fetchData();
//   }
//
//
//   /* NOT NEEDED YET */
//   /* Function to actually fetch data */
//   fetchData() {
//     fetch(REQUEST_URL)
//       .then((response) => response.json())
//       .then((responseData) => {
//         this.setState({
//           graph: responseData.graph,
//         });
//       })
//       .done();
//   }
//
//   /* Master render function. Will render image or loading screen if
//      trouble pinging server. */
//   render() {
//     /* Dislpays loading text if data not loaded */
//     /* NOT NEEDED
//     if (!this.state.loaded) {
//       return this.renderLoadingView();
//     } */
//     return (
//       <View style={styles.container}>
//         <RNChart style={styles.chart}       // Render the graph
//                  chartData={chartData}
//                  verticalGridStep={5}
//                  xLabels={xLabels}
//              />
//         <Animated.Image                    // Render the image
//                  source={{uri: commURL}}
//                  style={styles.image}
//              />
//       </View>
//     );
//   }
//
//   /* Renders loading text */
//   /* CURRENTLY NOT USED */
//   renderLoadingView() {
//     return (
//       <View style={styles.container}>
//         <Text>
//           Loading data...
//         </Text>
//       </View>
//     );
//   }
//
//   /* Renders the graph */
//   renderGraph() {
//     return (
//       <View style={styles.container}>
//         <RNChart style={styles.chart}
//                  chartData={chartData}
//                  verticalGridStep={5}
//                  xLabels={xLabels}
//              />
//         </View>
//     );
//   }
//
//   /* Render the image */
// /*  renderImage(): ReactElement {
//     return (
//       <Animated.Image                         // Base: Image, Text, View
//         source={{uri: 'http://i.imgur.com/XMKOH81.jpg'}}
//         style={styles.image}
//       />
//     );
//   } */
// }
//
// AppRegistry.registerComponent('dashboard', () => dashboard);
