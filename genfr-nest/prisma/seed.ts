import { PrismaClient } from "@prisma/client";
import * as bcrypt from "bcrypt";

const prisma = new PrismaClient();

const main = async () => {
  const bc = await bcrypt.hash("password", 10);
  const User = await prisma.user.upsert({
    where: { email: "KGHMPolskaMiedz@kghm.com" },
    update: {},
    create: {
      email: "KGHMPolskaMiedz@kghm.com",
      name: "John Doe",
      password: bc,
      is_enabled: true,
      role: "USER",
      isVerified:true
    },
  });

  const Admin = await prisma.user.upsert({
    where: { email: "patmikdev@gmail.com" },
    update: {},
    create: {
      email: "patmikdev@gmail.com",
      name: "Admin",
      password: bc, 
      is_enabled: true,
      role: "ADMIN",
      isVerified:true
    },
  });

  const Tripcord = await prisma.user.upsert({
    where: { email: "tpc@gmail.com" },
    update: {},
    create: {
      email: "tpc@gmail.com",
      name: "Trip Cord",
      password: bc,
      is_enabled: true,
      role: "USER",
      isVerified:true
    },
  });


  console.warn(User, "User created successfully");
  console.warn(Admin, "Admin created successfully");
  console.warn(Tripcord, "Tripcord created successfully");
  console.warn("Seed completed successfully");
};
main().catch((error: unknown) => {
  console.warn("Error While generating Seed: \n", error);
});
