import { StatusBar } from 'expo-status-bar';
import { useState } from 'react';
import { StyleSheet, Text, View,Button,SafeAreaView,Image } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { TouchableOpacity } from 'react-native';

export default function App() {
  const [image, setImage] = useState(null)
  const [result,setResult] = useState(null)
  

  const CameraPicker=async ()=>{
    setResult('')
    let result=await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      aspect:[4,4],
      quality:1,
      allowsEditing:true
    });
    console.log(result)
    if(!result.canceled){
      setImage(result.uri);
    }

  }

  const GalleryPicker= async ()=>{
    setResult('')
    let result=await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      aspect:[4,4],
      quality:1,
      allowsEditing:true
    });
    console.log(result)
    if(!result.canceled){
      setImage(result.uri);
    }

  };
  
  const send= async()=>{
    let data = new FormData();
    data.append('image',{
      uri:image,
      type:'image/jpeg',
      name:'image.jpg'
    })
    console.log("");
    fetch('https://0f86-2401-4900-614e-9a5d-8d63-2565-80c3-102.ngrok-free.app/process',{
      method:'POST',
      body:data,
      headers:{
        'Content-Type':'multipart/form-data'
      }
    })
    .then(res=>res.json())
    .then(d=>{
      setResult(d)
      console.log(d);
    })
  }

  const vehicleFound=()=>{
    if(result.detected_color==result.original_color && result.detected_type==result.original_type)
    {
      var out="The vehicle is genuine.";
    }
    else if(result.detected_color!=result.original_color && result.detected_type==result.original_type)
    {
      var out="The vehicle is fake since the original color is "+result.original_color+".";
    }
    else if(result.detected_color==result.original_color && result.detected_type!=result.original_type)
    {
      var out="The vehicle is fake since the original model is "+result.original_type+".";
    }
    else if(result.detected_color!=result.original_color && result.detected_type!=result.original_type)
    {
      var out="The vehicle is fake since the original color is "+result.original_color+" and the original model is "+result.original_type+".";
    }
    return out;
  }

  const vehicleNotFound=()=>{
    var out="The vehicle is fake. The vehicle is not found in the database.";
    return out;
  }
  
  return (
    
    <SafeAreaView style={{backgroundColor:'#fff',alignItems:'center',flex:1}}>
      <View style={styles.appbar}>
        <Text style={{color:'#fff',fontSize:36,fontWeight:'bold'}}>VehiScan</Text>
      </View>
      <Text style={{color:'#000',fontSize:30,fontWeight:'bold',paddingTop:30,paddingBottom:30}}>Check Your Vehicle!</Text>
      <View style={{ flex: 1, alignItems: 'center',gap:10}}>
        {image?
          <Image source={{ uri: image}} style={{ width: 250, height: 250 }}/>
          :
          <Image source={require('./assets/selectimage.png')} style={{ width: 250, height: 250 }}/>
        }
        <View style={{marginTop:20,gap:15,flexDirection:'row'}}>
          <Button title='Camera Image' onPress={CameraPicker}></Button>
          <Button title='Gallery Image' onPress={GalleryPicker}></Button>
        </View>
          <Button title='Check Vehicle' color={'red'} onPress={send}> </Button>
        
        <View style={styles.outputbox}>
          {result?<>
          {result.status=="true"?<Text style={styles.outputtext}>{vehicleFound()}</Text>:<Text style={styles.outputtext}>{vehicleNotFound()}</Text>}
          </>
          :<>
          </>}

        </View>
      </View>
      
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  appbar:{
    width:'100%',
    height:100,
    backgroundColor:'#065465',
    alignItems:'center',
    justifyContent:'center',
    paddingTop:40,  
  },
  outputbox:{
    // borderWidth:0.5,
    borderColor:'#888',
    width:330,
    height:150,
    backgroundColor:'#fff',
    marginTop:15,
    borderRadius:10,
    alignItems:'center',
    justifyContent:'center',
  },
  outputtext:{
    fontSize:24,
    fontWeight:'bold',
    padding:10,
    color:'red',
    textAlign:'center'

  }

});
