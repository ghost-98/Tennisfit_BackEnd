import 'package:flutter/material.dart';
import 'package:tennis_swing_similarity/pages/tennis_swing_similarity.dart';

void main() async{
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
          fontFamily: "PretendardBold"
      ),
      home: TennisSwingSimilarity(),
    );
  }
}
