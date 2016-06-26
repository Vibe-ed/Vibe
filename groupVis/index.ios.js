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
import Svg,{
    Circle,
    Ellipse,
    G,
    LinearGradient,
    RadialGradient,
    Line,
    Path,
    Polygon,
    Polyline,
    Rect,
    Symbol,
    Use,
    Defs,
    Stop
} from 'react-native-svg';

var graphAPI1 = {
  graph:  [[1, 1, 0],
           [2, 1, 1],
           [0, 0, 1]]
}

var graphAPI2 = {
  users: ['henry', 'yael', 'tyler'],
  numUsers: 3
}

var styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'white',
    },
    header: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      height: 40,
      backgroundColor: 'black',
    },
    title: {
      position: 'absolute',
      top: 2,
      left: 125,
      right: 1,
      height: 30,
      backgroundColor: 'black',
    },
    iconUser0: {
      height: 60 + graphAPI1.graph[0][0] * 10,
      width: 60 + graphAPI1.graph[0][0] * 10,
      // left: 20,
      right: 125,
      top: 60,
      position: 'absolute',
    },
    iconUser1: {
      height: 60 + graphAPI1.graph[1][1] * 10,
      width: 60 + graphAPI1.graph[1][1] * 10,
      left: 20,
      right: 275,
      bottom: 80,
      position: 'absolute',
      tintColor: 'red'
    },
    iconUser2: {
      height: 60 + graphAPI1.graph[2][2] * 10,
      width: 60 + graphAPI1.graph[2][2] * 10,
      left: 225,
      right: 200,
      bottom: 80,
      position: 'absolute',
    },
    arrowTail0_1: {
      backgroundColor: 'transparent',
      position: 'absolute',
      borderBottomColor: 'transparent',
      borderLeftColor: 'transparent',
      borderRightColor: 'transparent',
      borderBottomWidth: 0,
      borderLeftWidth: 0,
      borderRightWidth: 0,
      borderTopWidth: 3,
      borderTopColor: 'red',
      borderStyle: 'solid',
      borderTopLeftRadius: 12,
      top: 215,
      left: -10,
      width: 200,
      height: 20,
      transform: [
        {rotate: '111deg'}
      ]
    },
    arrowHead0_1: {
      backgroundColor: 'transparent',
      width: 0,
      height: 0,
      borderTopWidth: 12,
      borderTopColor: 'transparent',
      borderRightWidth: 12,
      borderRightColor: 'red',
      borderStyle: 'solid',
      transform: [
        {rotate: '68deg'}
      ],
      position: 'absolute',
      bottom: 156,
      left: 58,
      overflow: 'visible'
    },
    arrowTail0_2: {
      backgroundColor: 'transparent',
      position: 'absolute',
      borderBottomColor: 'transparent',
      borderLeftColor: 'transparent',
      borderRightColor: 'transparent',
      borderBottomWidth: 0,
      borderLeftWidth: 0,
      borderRightWidth: 0,
      borderTopWidth: 3,
      borderTopColor: 'red',
      borderStyle: 'solid',
      borderTopLeftRadius: 12,
      top: 220,
      right: 10,
      width: 200,
      height: 20,
      transform: [
        {rotate: '69deg'}
      ]
    },
    arrowHead0_2: {
      backgroundColor: 'transparent',
      width: 0,
      height: 0,
      borderTopWidth: 12,
      borderTopColor: 'transparent',
      borderRightWidth: 12,
      borderRightColor: 'red',
      borderStyle: 'solid',
      transform: [
        {rotate: '20deg'}
      ],
      position: 'absolute',
      bottom: 156,
      right: 61,
      overflow: 'visible'
    },
    arrowTail1_0: {
      backgroundColor: 'transparent',
      position: 'absolute',
      borderBottomColor: 'transparent',
      borderLeftColor: 'transparent',
      borderRightColor: 'transparent',
      borderBottomWidth: 0,
      borderLeftWidth: 0,
      borderRightWidth: 0,
      borderTopWidth: 3,
      borderTopColor: 'red',
      borderStyle: 'solid',
      borderTopLeftRadius: 12,
      top: 230,
      left: 15,
      width: 200,
      height: 20,
      transform: [
        {rotate: '291deg'}
      ]
    },
    arrowHead1_0: {
      backgroundColor: 'transparent',
      width: 0,
      height: 0,
      borderTopWidth: 12,
      borderTopColor: 'transparent',
      borderRightWidth: 12,
      borderRightColor: 'red',
      borderStyle: 'solid',
      transform: [
        {rotate: '248deg'}
      ],
      position: 'absolute',
      top: 138,
      left: 136,
      overflow: 'visible'
    },
    arrowTail1_2: {
      backgroundColor: 'transparent',
      position: 'absolute',
      borderBottomColor: 'transparent',
      borderLeftColor: 'transparent',
      borderRightColor: 'transparent',
      borderBottomWidth: 0,
      borderLeftWidth: 0,
      borderRightWidth: 0,
      borderTopWidth: 3,
      borderTopColor: 'red',
      borderStyle: 'solid',
      borderTopLeftRadius: 12,
      bottom: 80,
      left: 75,
      width: 160,
      height: 20,
      transform: [
        {rotate: '0deg'}
      ]
    },
    arrowHead1_2: {
      backgroundColor: 'transparent',
      width: 0,
      height: 0,
      borderTopWidth: 12,
      borderTopColor: 'transparent',
      borderRightWidth: 12,
      borderRightColor: 'red',
      borderStyle: 'solid',
      transform: [
        {rotate: '-45deg'}
      ],
      position: 'absolute',
      bottom: 92,
      right: 80,
      overflow: 'visible'
    },
    arrowTail2_0: {
      backgroundColor: 'transparent',
      position: 'absolute',
      borderBottomColor: 'transparent',
      borderLeftColor: 'transparent',
      borderRightColor: 'transparent',
      borderBottomWidth: 0,
      borderLeftWidth: 0,
      borderRightWidth: 0,
      borderTopWidth: 3,
      borderTopColor: 'red',
      borderStyle: 'solid',
      borderTopLeftRadius: 12,
      top: 220,
      right: 3,
      width: 200,
      height: 20,
      transform: [
        {rotate: '-111deg'}
      ]
    },
    arrowHead2_0: {
      backgroundColor: 'transparent',
      width: 0,
      height: 0,
      borderTopWidth: 12,
      borderTopColor: 'transparent',
      borderRightWidth: 12,
      borderRightColor: 'red',
      borderStyle: 'solid',
      transform: [
        {rotate: '-160deg'}
      ],
      position: 'absolute',
      top: 135,
      right: 140,
      overflow: 'visible'
    },
    arrowTail2_1: {
      backgroundColor: 'transparent',
      position: 'absolute',
      borderBottomColor: 'transparent',
      borderLeftColor: 'transparent',
      borderRightColor: 'transparent',
      borderBottomWidth: 0,
      borderLeftWidth: 0,
      borderRightWidth: 0,
      borderTopWidth: 3,
      borderTopColor: 'red',
      borderStyle: 'solid',
      borderTopLeftRadius: 12,
      bottom: 110,
      right: 75,
      width: 160,
      height: 20,
      transform: [
        {rotate: '180deg'}
      ]
    },
    arrowHead2_1: {
      backgroundColor: 'transparent',
      width: 0,
      height: 0,
      borderTopWidth: 12,
      borderTopColor: 'transparent',
      borderRightWidth: 12,
      borderRightColor: 'red',
      borderStyle: 'solid',
      transform: [
        {rotate: '135deg'}
      ],
      position: 'absolute',
      bottom: 106,
      left: 80,
      overflow: 'visible'
    }
});

var graphLabels = {
  users: graphAPI2.users
}

var commURL = 'https://d30y9cdsu7xlg0.cloudfront.net/png/5024-200.png'
class groupVis extends React.Component {
  /* Master render function. Will render image or loading screen if
     trouble pinging server. */
  render() {
    return (
      <View style={styles.container}>
        <Image                    // Render the image
               source={{uri: commURL}}
               style={styles.header}
             />
        <Image                    // Render the image
               source={require('./vibe_logo_simple.png')}
               style={styles.title}
             />
        <Animated.Image                    // Render the image
                 source={{uri: commURL}}
                 style={styles.iconUser0}
             />
        <Animated.Image                    // Render the image
                 source={{uri: commURL}}
                 style={styles.iconUser1}
             />
        <Animated.Image                    // Render the image
                 source={{uri: commURL}}
                 style={styles.iconUser2}
             />
        <View style={styles.arrowTail0_2} />
        <View style={styles.arrowHead0_2} />
        <View style={styles.arrowTail2_0} />
        <View style={styles.arrowHead2_0} />
        <View style={styles.arrowTail0_1} />
        <View style={styles.arrowHead0_1} />
        <View style={styles.arrowTail1_0} />
        <View style={styles.arrowHead1_0} />
        <View style={styles.arrowTail1_2} />
        <View style={styles.arrowHead1_2} />
        <View style={styles.arrowTail2_1} />
        <View style={styles.arrowHead2_1} />
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

AppRegistry.registerComponent('groupVis', () => groupVis);
